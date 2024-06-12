from typing import Optional


def extract_draft(response: Optional[str]) -> Optional[str]:
    match_str = "final answer:"
    if response:
        position = response.lower().find(match_str)
        if position == -1:
            return None
        else:
            return response[position + len(match_str) :].strip()
    else:
        return None
