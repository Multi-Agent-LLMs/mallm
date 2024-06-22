import os
import pprint
import sys
from contextlib import AbstractContextManager
from typing import Any, TextIO


class SuppressOutput:
    def __init__(self) -> None:
        self._original_stdout: TextIO = sys.stdout
        self._original_stderr: TextIO = sys.stderr

    def __enter__(self) -> None:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def suppress_output() -> AbstractContextManager[None]:
    return SuppressOutput()


def pretty_print_dict(
    config_dict: dict[str, Any], indent: int = 4, width: int = 80
) -> None:
    """
    Pretty prints a dictionary in a readable and styled format.

    Parameters:
    config_dict (dict): The dictionary to be pretty printed.
    indent (int): The number of spaces to use for indentation.
    width (int): The maximum number of characters per line.
    """
    print("\n" + "=" * width)
    print("CONFIGURATION PARAMETERS".center(width))
    print("=" * width + "\n")

    pp = pprint.PrettyPrinter(indent=indent, width=width)
    pp.pprint(config_dict)

    print("\n" + "=" * width)
    print("END OF CONFIGURATION PARAMETERS".center(width))
    print("=" * width + "\n")
