from __future__ import annotations

import httpx
import logging
from typing import Any, TYPE_CHECKING, Optional

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.evaluation.evaluator import Evaluator
if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
from mallm.agents.agent import Agent

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
        references: list[str] = [],
    ):
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
        self.judgements = []
        self.intervention_type = intervention_type
        self.coordinator = coordinator
        self.references = references
    
    def intervention(self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        answer: str) -> bool:
        
        self.judgements.append(Evaluator.calculate_score(answer, self.references, self.metric))
        logger.debug(f"Judge's performances: {self.judgements}")

        if len(self.judgements) > 1 and self.judgements[-1] < self.judgements[-2]:  # regenerates at most once per turn

            if self.intervention_type == "regenerate":
                # delete and restart the turn
                logger.debug("Judge decided to regenerate the turn.")
                return unique_id-len(self.coordinator.agents), True
            elif self.intervention_type == "policy":
                # Give the agents tips on how to improve their policy
                logger.debug("Judge decided to give policy feedback.")
                response = self.response_generator.generate_policy_intervention(
                    template_filling,
                    provide_labels=False
                )
                logger.debug(f"Judge's policy feedback: {response.message}")
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
                return unique_id+1, False
        return unique_id, False