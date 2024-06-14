from typing import Optional


def extract_draft(response: Optional[str]) -> Optional[str]:
    match_str = "final answer"
    if response:
        position = response.lower().find(match_str)
        if position == -1:
            return None
        else:
            matched_str = response[position + len(match_str) :].strip()
            if matched_str[0] == "]" or matched_str[0] == ":":
                matched_str = matched_str[1:].strip()
            return matched_str
    else:
        return None
