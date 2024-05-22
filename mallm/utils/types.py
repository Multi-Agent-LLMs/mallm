from dataclasses import dataclass
from typing import Optional


@dataclass
class Agreement:
    agreement: Optional[bool]
    response: str
    agent_id: str
    persona: str


@dataclass
class Memory:
    pass


@dataclass
class TemplateFilling:
    task_instruction: str
    input_str: str
    current_draft: Optional[str]
    persona: str
    persona_description: str
    agent_memory: Optional[list[dict[str, str]]]
    sents_min: Optional[int] = None
    sents_max: Optional[int] = None
