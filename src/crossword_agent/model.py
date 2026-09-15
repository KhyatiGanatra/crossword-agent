"""Model access for the sweep-fill solver.

The solver asks the model exactly one kind of question ("what could these clues be?") and expects
a JSON object back. This module hides the provider details behind ``AnswerModel``:

- ``NebiusModel`` calls Token Factory chat completions with ``response_format: json_object``,
  reasoning switched off unless the caller asks for a bounded reasoning pass, prices from the live
  catalog, and a clear split between recoverable failures (returned in ``JsonReply.error``) and
  configuration failures (raised as ``ModelError``).
- ``FixtureModel`` is the offline test double: answers come from the fixture's candidate bank.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib import error, request

from .types import JsonReply


class ModelError(RuntimeError):
    """Provider failure. Recoverable ones (timeouts, 429/5xx) skip a batch; others stop the run."""

    def __init__(self, message: str, *, category: str = "model", recoverable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.recoverable = recoverable


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in ``text``, tolerating prose before it and junk after it."""

    start = text.find("{")
    if start < 0:
        return None
    candidate = text[start:]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        end = candidate.rfind("}")
        if end < 0:
            return None
        try:
            data = json.loads(candidate[: end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


class AnswerModel(Protocol):
    @property
    def name(self) -> str: ...

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        max_tokens: int = 4000,
        model: str | None = None,
    ) -> JsonReply: ...


_CLUE_LINE = re.compile(r"^(\d+[AD]) \(\d+\)", re.MULTILINE)


@dataclass
class FixtureModel:
    """Offline test double: every clue id found in the prompt is answered from the candidate bank."""

    candidate_bank: dict[str, tuple[str, ...]]
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fixture-candidate-bank"

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        max_tokens: int = 4000,
        model: str | None = None,
    ) -> JsonReply:
        self.calls += 1
        self.prompts.append(user)
        ids = _CLUE_LINE.findall(user)
        answers = {entry_id: list(self.candidate_bank.get(entry_id, ())) for entry_id in ids}
        return JsonReply(data={"answers": answers}, model=model or self.name)


_NON_RECOVERABLE_HTTP = frozenset({400, 401, 403, 404, 405, 413, 415, 422})


class NebiusModel:
    """Dependency-free Token Factory chat-completions adapter."""

    def __init__(
        self,
        model: str = "moonshotai/Kimi-K3",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("NEBIUS_API_KEY")
        self._base_url = (
            base_url or os.getenv("NEBIUS_BASE_URL") or "https://api.tokenfactory.nebius.com/v1"
        ).rstrip("/")
        self._timeout_seconds = timeout_seconds
        if not self._api_key:
            raise ModelError("NEBIUS_API_KEY is required for the Nebius provider")
        self._catalog = self._load_catalog()

    @property
    def name(self) -> str:
        return self._model

    def price(self, model_id: str) -> tuple[float, float] | None:
        entry = self._catalog.get(model_id)
        if entry is None:
            return None
        return entry["prompt"], entry["completion"]

    def supports_reasoning(self, model_id: str) -> bool:
        entry = self._catalog.get(model_id)
        # Unknown model (or catalog unavailable): assume a reasoning-capable model so that
        # thinking is switched off explicitly; most current Token Factory models reason by default.
        return entry is None or "reasoning" in entry["features"]

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        max_tokens: int = 4000,
        model: str | None = None,
    ) -> JsonReply:
        model_id = model or self._model
        body: dict[str, Any] = {
            "model": model_id,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.supports_reasoning(model_id):
            # "none" is the portable off-switch on Token Factory; GLM ignores chat_template_kwargs
            # but honours it, and DeepSeek honours both.
            body["reasoning_effort"] = "low" if thinking else "none"
            if not thinking and "deepseek" in model_id.lower():
                body["chat_template_kwargs"] = {"thinking": False}

        try:
            payload = self._request(body)
        except ModelError as exc:
            if not exc.recoverable:
                raise
            return JsonReply(data=None, model=model_id, error=f"{exc.category}: {exc}")
        input_tokens, output_tokens, reasoning_tokens, cost = self._usage(payload, model_id)
        choice = (payload.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        content = (choice.get("message") or {}).get("content") or ""
        data = parse_json_object(content)
        error_text = None
        if data is None:
            error_text = (
                f"no JSON object in the response (finish_reason={finish_reason}, "
                f"reasoning_tokens={reasoning_tokens})"
            )
        return JsonReply(
            data=data,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            estimated_cost_usd=cost,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            error=error_text,
        )

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST once, retry once on transient failures, raise ModelError otherwise."""

        encoded = json.dumps(body).encode("utf-8")
        last_error: ModelError | None = None
        for attempt in range(2):
            http_request = request.Request(
                f"{self._base_url}/chat/completions",
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(http_request, timeout=self._timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in _NON_RECOVERABLE_HTTP:
                    raise ModelError(
                        f"Nebius returned HTTP {exc.code}: {detail}", category="http"
                    ) from exc
                last_error = ModelError(
                    f"Nebius returned HTTP {exc.code}: {detail}", category="http", recoverable=True
                )
            except TimeoutError as exc:
                last_error = ModelError(
                    f"Nebius request timed out after {self._timeout_seconds:g} seconds: {exc}",
                    category="timeout",
                    recoverable=True,
                )
            except error.URLError as exc:
                category = "timeout" if isinstance(exc.reason, TimeoutError) else "transport"
                last_error = ModelError(
                    f"Nebius request failed: {exc.reason}", category=category, recoverable=True
                )
            except json.JSONDecodeError as exc:
                last_error = ModelError(
                    f"Nebius returned a non-JSON response: {exc}",
                    category="response_parse",
                    recoverable=True,
                )
            if attempt == 0:
                time.sleep(1.0)
        assert last_error is not None
        raise last_error

    def _usage(
        self, payload: dict[str, Any], model_id: str
    ) -> tuple[int, int, int, float | None]:
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(
            usage.get("reasoning_tokens", details.get("reasoning_tokens", 0)) or 0
        )
        price = self.price(model_id)
        cost = None
        if price is not None:
            cost = input_tokens * price[0] + output_tokens * price[1]
        return input_tokens, output_tokens, reasoning_tokens, cost

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        """Read per-token prices and feature flags for every model (free GET); empty on failure."""

        catalog_request = request.Request(
            f"{self._base_url}/models?verbose=true",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            with request.urlopen(catalog_request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError):
            return {}
        catalog: dict[str, dict[str, Any]] = {}
        for item in payload.get("data", []):
            pricing = item.get("pricing") or {}
            try:
                catalog[str(item["id"])] = {
                    "prompt": float(pricing["prompt"]),
                    "completion": float(pricing["completion"]),
                    "features": frozenset(item.get("supported_features") or ()),
                }
            except (KeyError, TypeError, ValueError):
                continue
        return catalog
