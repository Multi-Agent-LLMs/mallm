# You can run a test by running the following command:
# python -m mallm.scripts.batch_mallm <your_config_file>

import re
from collections.abc import Iterator
from typing import Any, Optional


class _Delta:
    def __init__(self, content: Optional[str]) -> None:
        self.content = content


class _LogprobToken:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class _Logprobs:
    def __init__(self, content: list[_LogprobToken]) -> None:
        self.content = content


class _Choice:
    def __init__(self, delta: _Delta, logprobs: Optional[_Logprobs]) -> None:
        self.delta = delta
        self.logprobs = logprobs


class _Chunk:
    def __init__(self, content: str) -> None:
        # mimic OpenAI's streaming chunk shape minimally
        self.choices = [
            _Choice(delta=_Delta(content=content), logprobs=_Logprobs([_LogprobToken(0.0)]))
        ]


def _extract_final_solution_from_messages(messages: list[dict[str, str]]) -> str:
    # Very lightweight heuristic to create deterministic outputs for tests
    last = messages[-1]["content"] if messages else ""
    joined = "\n".join(m.get("content", "") for m in messages)

    # Explicit verdict patterns
    if "[[A]]" in last or "final verdict" in last:
        return "[[A]]"

    # Ranking, voting, and numeric-only prompts
    if "provide the number of the solution" in last.lower():
        return "0"
    if "please provide the numbers of the solutions" in last.lower():
        return "0"
    if "distribute 10 points" in last.lower():
        return '{"0": 10}'
    if "provide the rankings" in last.lower():
        return "0 1 2"
    if "generate a confidence score" in last.lower():
        return "100"

    # Challenge / agreement prompts
    if "respond with the exact word 'agree'" in last.lower():
        return "AGREE"
    if "[agree]" in last.lower():
        return "[AGREE]"

    # Extraction prompts
    if "extract the final solution" in last.lower():
        # Prefer extracting from the message that actually contains "Your previous response:"
        src = ""
        for msg in reversed(messages):
            content = msg.get("content", "") or ""
            if re.search(r"\bprevious response:\b", content, re.IGNORECASE):
                src = content
                break
        if not src:
            src = last

        m = re.search(r"previous response:\s*(.*)$", src, re.IGNORECASE | re.DOTALL)
        extracted = (m.group(1) if m else src).strip()

        # If the extraction instruction was merged into the same user message, strip it off.
        cut = re.search(r"\n\s*extract the final solution\b", extracted, re.IGNORECASE)
        if cut:
            extracted = extracted[: cut.start()].strip()

        # If the previous response contains a "Final Solution:" marker, return only the solution part.
        m2 = re.search(r"final solution:\s*(.*)$", extracted, re.IGNORECASE | re.DOTALL)
        if m2:
            return m2.group(1).strip()

        return extracted or "Final Solution"

    # Task-specific simple heuristics
    if "capital of france" in joined.lower():
        return "AGREE. Final Solution: Paris"
    if "6 + 7" in joined or "6+7" in joined:
        return "AGREE. Final Solution: 13"

    # Default generic response
    return "AGREE. Final Solution: OK"


class _Completions:
    @staticmethod
    def create(
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        stop: Optional[list[str]] = None,
        max_tokens: int = 1024,
        logprobs: bool = True,
        **kwargs: Any,
    ) -> Iterator[_Chunk]:
        content = _extract_final_solution_from_messages(messages)

        def _generator() -> Iterator[_Chunk]:
            # yield a single chunk; the caller aggregates chunks into a string
            yield _Chunk(content)

        return _generator()


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class MockOpenAI:
    """
    Minimal mock of the OpenAI client that exposes chat.completions.create
    and yields streaming-like chunks compatible with the code paths used in Chat._call.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _Chat()
