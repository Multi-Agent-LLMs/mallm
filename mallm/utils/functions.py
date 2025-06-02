from typing import Optional


def extract_draft(response: Optional[str], match_str: str = "final solution") -> Optional[str]:
    if response:
        position = response.lower().rfind(match_str)
        if position == -1:
            return None
        matched_str = response[position + len(match_str) :].strip()
        if matched_str[0] == "]" or matched_str[0] == ":":
            matched_str = matched_str[1:].strip()
        return matched_str # because LM tends to add extra explanation afterwards
    return None
