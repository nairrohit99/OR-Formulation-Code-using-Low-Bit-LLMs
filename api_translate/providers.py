"""Thin wrappers around the Claude (Anthropic SDK) and Gemini (google-genai SDK) APIs.

Each provider turns a prompt string into a `Completion`. API keys come from the environment:
  Claude: ANTHROPIC_API_KEY (or an `ant auth login` profile)
  Gemini: GEMINI_API_KEY (or GOOGLE_API_KEY)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

# Models that accept the server-side refusal fallback parameter (fallbacks="default").
CLAUDE_FALLBACK_MODELS = {"claude-opus-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Completion:
    text: str = ""
    ok: bool = True
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict = field(default_factory=dict)


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 16000,
                 use_fallbacks: bool = True, max_retries: int = 6):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries)  # SDK retries 429/5xx with backoff
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.use_fallbacks = use_fallbacks and model in CLAUDE_FALLBACK_MODELS

    def _params(self, prompt: str) -> dict:
        p = dict(model=self.model, max_tokens=self.max_tokens, messages=[{"role": "user", "content": prompt}])
        if self.effort:
            p["output_config"] = {"effort": self.effort}
        return p

    def _to_completion(self, msg) -> Completion:
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        c = Completion(text=text, input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                       meta={"stop_reason": msg.stop_reason, "model": msg.model})
        if msg.stop_reason == "refusal":
            c.ok, c.error = False, "refusal"
        elif msg.stop_reason == "max_tokens":
            c.meta["truncated"] = True
        return c

    def complete(self, prompt: str) -> Completion:
        a = self.anthropic
        try:
            if self.use_fallbacks:
                msg = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **self._params(prompt))
            else:
                msg = self.client.messages.create(**self._params(prompt))
        except a.BadRequestError as e:          # not retryable: report and move on
            return Completion(ok=False, error=f"bad_request: {e.message}")
        except a.AuthenticationError:
            raise SystemExit("Claude authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`.")
        except a.PermissionDeniedError as e:
            raise SystemExit(f"Claude permission denied: {e.message}")
        except a.NotFoundError:
            raise SystemExit(f"Claude model not found: {self.model}")
        except a.RateLimitError as e:           # SDK already retried
            return Completion(ok=False, error=f"rate_limited: {e.message}")
        except a.APIStatusError as e:
            return Completion(ok=False, error=f"api_status_{e.status_code}: {e.message}")
        except a.APIConnectionError as e:
            return Completion(ok=False, error=f"connection: {e}")
        return self._to_completion(msg)

    # ---- Message Batches: 50% cheaper, asynchronous (usually < 1 h). `fallbacks` is not allowed here.
    def submit_batch(self, requests: list[tuple[str, str]]) -> str:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = self.client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**self._params(prompt)))
            for cid, prompt in requests
        ])
        return batch.id

    def wait_batch(self, batch_id: str, poll_seconds: int = 60) -> dict[str, Completion]:
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            rc = b.request_counts
            print(f"  batch {batch_id}: processing {rc.processing}, succeeded {rc.succeeded}, errored {rc.errored}", flush=True)
            time.sleep(poll_seconds)
        out = {}
        for r in self.client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                out[r.custom_id] = self._to_completion(r.result.message)
            elif r.result.type == "errored":
                err = r.result.error
                etype = getattr(getattr(err, "error", None), "type", None) or getattr(err, "type", "unknown")
                out[r.custom_id] = Completion(ok=False, error=f"batch_errored: {etype}")
            else:
                out[r.custom_id] = Completion(ok=False, error=f"batch_{r.result.type}")  # canceled / expired
        return out


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, max_tokens: int = 16000, temperature: Optional[float] = None,
                 max_retries: int = 6):
        from google import genai
        from google.genai import errors, types
        self.genai, self.types, self.errors = genai, types, errors
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self.model, self.max_tokens, self.temperature, self.max_retries = model, max_tokens, temperature, max_retries

    def complete(self, prompt: str) -> Completion:
        cfg = {"max_output_tokens": self.max_tokens}
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.types.GenerateContentConfig(**cfg))
                um = getattr(resp, "usage_metadata", None)
                text = resp.text or ""
                c = Completion(text=text, input_tokens=getattr(um, "prompt_token_count", 0) or 0,
                               output_tokens=getattr(um, "candidates_token_count", 0) or 0)
                if not text:
                    fr = resp.candidates[0].finish_reason if resp.candidates else "no_candidates"
                    c.ok, c.error = False, f"empty_response: {fr}"
                return c
            except self.errors.ClientError as e:
                code = getattr(e, "code", None)
                if code in (401, 403):
                    raise SystemExit(f"Gemini authentication/permission error: {e}. Set GEMINI_API_KEY.")
                if code == 404:
                    raise SystemExit(f"Gemini model not found: {self.model}")
                if code != 429 or attempt == self.max_retries:
                    return Completion(ok=False, error=f"client_error_{code}: {e}")
            except self.errors.ServerError as e:
                if attempt == self.max_retries:
                    return Completion(ok=False, error=f"server_error: {e}")
            except Exception as e:  # network errors etc.
                if attempt == self.max_retries:
                    return Completion(ok=False, error=f"{type(e).__name__}: {e}")
            time.sleep(min(60, 2 ** attempt + random.random()))
        return Completion(ok=False, error="retries_exhausted")


def make_provider(provider: str, model: Optional[str], effort: str, max_tokens: int, use_fallbacks: bool,
                  temperature: Optional[float]):
    if provider == "claude":
        return ClaudeProvider(model=model or "claude-opus-5-5", effort=effort, max_tokens=max_tokens,
                              use_fallbacks=use_fallbacks)
    if provider == "gemini":
        if not model:
            raise SystemExit("--model is required for Gemini (use the exact model id from Google AI Studio).")
        return GeminiProvider(model=model, max_tokens=max_tokens, temperature=temperature)
    raise SystemExit(f"unknown provider {provider}")
