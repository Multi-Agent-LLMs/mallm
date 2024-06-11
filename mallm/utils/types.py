from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Agreement:
    agreement: Optional[bool]
    response: str
    agent_id: str
    persona: str


@dataclass
class Memory:
    message_id: int
    turn: int
    agent_id: str
    persona: str
    contribution: str
    text: str
    agreement: Optional[bool]
    extracted_draft: Optional[str]
    memory_ids: list[int]
    additional_args: dict[str, Any]


@dataclass
class TemplateFilling:
    task_instruction: str
    input_str: str
    current_draft: Optional[str]
    persona: str
    persona_description: str
    agent_memory: Optional[list[dict[str, str]]]
    feedback_sentences: Optional[tuple[int, int]] = None


@dataclass
class InputExample:
    example_id: str
    dataset_id: Optional[str]
    inputs: list[str]
    context: Optional[list[str]]
    references: list[str]
    personas: Optional[list[str]]

    def confirm_types(self) -> None:
        # Confirm type of example_id
        assert isinstance(self.example_id, str)
        # Confirm type of dataset_id
        if self.dataset_id is not None:
            assert isinstance(self.dataset_id, str)
        # Confirm type of input
        assert isinstance(self.inputs, list)
        # Confirm type of references
        assert isinstance(self.references, list)
        # Confirm type of personas
        if self.personas is not None:
            assert isinstance(self.personas, list)
            for p in self.personas:
                assert isinstance(p, str)
        # Confirm type of context
        if self.context is not None:
            assert isinstance(self.context, list)
            for c in self.context:
                assert isinstance(c, str)
        # Confirm type of references
        for r in self.references:
            assert isinstance(r, str)
