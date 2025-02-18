from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import httpx

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator

from mallm.agents.agent import Agent
from mallm.evaluation.evaluator import Evaluator
from mallm.utils.types import Memory, TemplateFilling

logger = logging.getLogger("mallm")


class Judge(Agent):
    def __init__(
        self,
        llm: Chat,
        client: httpx.Client,
        coordinator: Coordinator,
        response_generator: ResponseGenerator,
        persona: str,
        persona_description: str,
        metric: str,
        chain_of_thought: bool = False,
        drafting_agent: bool = False,
        intervention_type: str = "regenerate",
        references: Optional[list[str]] = None,
    ):
        if references is None:
            references = []
        super().__init__(
            llm,
            client,
            coordinator,
            response_generator,
            persona,
            persona_description,
            chain_of_thought,
            drafting_agent,
        )
        self.metric = Evaluator._initialize_metrics([metric])[0]
        self.judgements: list[Optional[bool]] = []
        self.performances: list[float] = []
        self.judged_solutions: list[str] = []
        self.intervention_type = intervention_type
        self.coordinator = coordinator
        self.references = references

    def llm_as_a_judge(self, template_filling: TemplateFilling) -> Optional[bool]:
        repeats = 0
        while repeats < 3:
            # check for drift
            response = self.response_generator.generate_judgement(
                template_filling, self.judged_solutions[-2], self.judged_solutions[-1]
            )
            if "[[A]]" in response.message:
                return True     # answer_before is better
            if "[[B]]" in response.message:
                return False    # answer_after is better (problem drift)
            logger.warning(f"Judge verdict is not valid: {response.message}. Retry number {repeats + 1}.")
            repeats += 1
        logger.warning(f"Judge verdict is not valid: {response.message}. All retries failed. The verdict will be saved as None.")
        return None

    def intervention(self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        answer: str,
        threshold: float = 0,
        always_intervene: bool = False,
        ) -> tuple[int, int]:
        self.judged_solutions.append(answer)

        if self.coordinator.judge_llm is not None:
            if len(self.judged_solutions) < 2:
                logger.debug("Judge skipped this turn because there are not enough solutions to judge.")
                return unique_id, turn
            on_track = self.llm_as_a_judge(template_filling)
        else:
            self.performances.append(Evaluator.calculate_score(answer, self.references, self.metric)["value"])
            on_track = len(self.performances) > 1 and self.performances[-1] + threshold < self.performances[-2]
        self.judgements.append(on_track)

        if on_track is False or always_intervene:  # regenerates at most once per turn
            if self.intervention_type == "regenerate":
                # delete and restart the turn
                logger.debug("Judge decided to regenerate the turn.")
                self.coordinator.forget_memories(turn)
                return unique_id - len(self.coordinator.agents) + 1, turn - 1
            if self.intervention_type == "policy":
                # Give the agents tips on how to improve their policy
                logger.debug("Judge decided to give policy feedback.")
                response = self.response_generator.generate_policy_intervention(
                    template_filling,
                    provide_labels=False
                )
                memory = Memory(
                    message_id=unique_id,
                    turn=turn,
                    agent_id=self.id,
                    persona=self.persona,
                    contribution="judge",
                    message=response.message,
                    agreement=None,
                    solution=None,
                    memory_ids=memory_ids,
                    additional_args={},
                )
                self.coordinator.update_memories([memory], self.coordinator.agents)
                self.coordinator.memory.append(memory)
                return unique_id + 1, turn
        return unique_id, turn
