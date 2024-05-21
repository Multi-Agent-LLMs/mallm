from dataclasses import dataclass
from typing import Optional


@dataclass
class Agreement:
    agreement: Optional[bool]
    response: str
    agent_id: str
    persona: str
