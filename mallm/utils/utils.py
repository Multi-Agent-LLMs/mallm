import os
import sys
from typing import Any, ContextManager, TextIO


class SuppressOutput:
    def __init__(self) -> None:
        self._original_stdout: TextIO = sys.stdout
        self._original_stderr: TextIO = sys.stderr

    def __enter__(self) -> None:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def suppress_output() -> ContextManager[None]:
    return SuppressOutput()
