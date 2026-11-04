"""Groq adapter (the PRD default provider).

The model id always comes from `.env` (`LLM_MODEL`), never from code or
`config.yaml`. The SDK is imported lazily so a keyless demo never needs the
package to be importable before `requirements.txt` is installed.
"""

from __future__ import annotations


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str):
        try:
            from groq import Groq
        except ImportError as exc:                 # pragma: no cover - env issue
            raise RuntimeError(
                "groq is not installed - run `pip install -r requirements.txt`"
            ) from exc

        self.model = model
        self._client = Groq(api_key=api_key)

    def complete(self, system: str, user: str, *, temperature: float,
                 max_tokens: int, timeout: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not response.choices:
            return ""
        return (response.choices[0].message.content or "").strip()
