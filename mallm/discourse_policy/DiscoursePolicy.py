from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
logger = logging.getLogger("mallm")


class DiscoursePolicy(ABC):

    @abstractmethod
    def discuss(
        self,
        coordinator: Coordinator,
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
