from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch import Tensor


@dataclass
class Agreement:
    agreement: Optional[bool]
    response: str
    solution: str
    agent_id: str
    persona: str
    message_id: int


@dataclass
class Response:
    agreement: Optional[bool]
    message: str
    solution: str


@dataclass
class Memory:
    message_id: int
    turn: int
    agent_id: str
    persona: str
    contribution: str
    message: str
    agreement: Optional[bool]
    solution: Optional[str]
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
class VotingResult:
    votes: Any
    final_answer: str
    most_voted: int
    agreed: bool


@dataclass
class VotingResultList:
    answers: list[str]
    type: str
    voting_process_string: str
    alterations: dict[str, VotingResult]


@dataclass
class InputExample:
    example_id: str
    dataset_id: Optional[str]
    inputs: list[str]
    context: Optional[list[str]]
    references: list[str]
    metadata: Optional[dict[Any, Any]] = None

    def confirm_types(self) -> None:
        assert isinstance(self.example_id, str), "Example_id is not a string"
        if self.dataset_id is not None:
            assert isinstance(self.dataset_id, str), "Dataset_id is not a string"
        assert isinstance(self.inputs, list), "Inputs is not a list"
        if self.metadata is not None:
            assert isinstance(self.metadata, dict), "Metadata is not a dict"
        for i in self.inputs:
            assert isinstance(i, str), "Inputs is not a list of only strings"
        assert isinstance(self.references, list), "References is not a list"
        for r in self.references:
            assert isinstance(r, str), "References is not a list of only strings"
        if self.context is not None:
            assert isinstance(self.context, list), "Context is not a list"
            for c in self.context:
                assert isinstance(c, str), "Context is not a list of only strings"
        for r in self.references:
            assert isinstance(r, str), "References is not a list of only strings"


@dataclass
class WorkerFunctions:
    worker_paraphrase_function: Callable[[list[str]], list[Tensor]]
    worker_context_function: Callable[[str], str]
    worker_persona_diversity_function: Callable[[list[str]], float]
