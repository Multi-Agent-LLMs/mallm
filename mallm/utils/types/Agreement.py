from dataclasses import dataclass


@dataclass
class Agreement:
    agreement: bool
    response: str
    agent_id: str
    persona: str
