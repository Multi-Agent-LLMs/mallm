from abc import ABC, abstractmethod


class DiscoursePolicy(ABC):

    @abstractmethod
    def discuss(
        self,
        coordinator,
        task_instruction: str,
        input: str,
        use_moderator: bool = False,
        feedback_sentences: list[int] = (3, 4),
        max_turns: int = None,
        context_length: int = 1,
        include_current_turn_in_memory: bool = False,
        extract_all_drafts: bool = False,
        debate_rounds: int = 1,
    ):
        pass
