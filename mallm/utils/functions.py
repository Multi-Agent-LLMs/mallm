from typing import Optional


def extract_draft(response: Optional[str]) -> Optional[str]:
    match_str = "final solution"
    if response:
        position = response.lower().rfind(match_str)
        if position == -1:
            return None
        else:
            matched_str = response[position + len(match_str) :].strip()
            if matched_str[0] == "]" or matched_str[0] == ":":
                matched_str = matched_str[1:].strip()
            matched_str = matched_str.split("\n\n")[
                0
            ]  # because LM tends to add extra explanation afterwards
            return matched_str
    else:
        return None
