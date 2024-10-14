from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import httpx

from mallm.agents.agent import Agent
from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.types import Agreement, Memory, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator

logger = logging.getLogger("mallm")


class PolicyFeedback(Agent):
    """
    Represents a PolicyFeedback agent, a type of Agent that ensures that the other agents adhere to a given policy.
    This agent does not contribute to solving the task.
    """
    def __init__(
        self,
        llm: Chat,
        client: httpx.Client,
        coordinator: Coordinator,
        response_generator: ResponseGenerator,
        persona: str = "Policy Moderator",
        persona_description: str = "A super-intelligent individual who has a neutral position at all times.",
        policy: str = "The discussion should remain on topic. Unneccesary changes to the solution should be avoided. The discussion should produce a minimal solution that fits the task requirements. Do not overcomplicate the solution.",
    ) -> None:
        """
        Initializes a PolicyFeedback agent with the necessary components.
        """
        self.policy = policy
        persona_description = persona_description + " It gives concrete suggestions about how to improve the discussion and follow this policy: " + self.policy
        super().__init__(
            llm, client, coordinator, response_generator, persona, persona_description
        )

    def policy_feedback(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agreements: list[Agreement],
    ) -> tuple[str, Memory, list[Agreement]]:
        """
        Agent provides feedback on how to improve discussion to fit the given policy.
        """
        logger.debug(
            f"Agent [bold blue]{self.short_id}[/] provides policy feedback to a solution."
        )
        instr_prompt = {
            "role": "user",
            "content": "Based on the provided task and discussion, carefully evaluate the course of discussion. Give concrete suggestions about how to improve the discussion and follow this policy: " + self.policy,
        }
        current_prompt = [
            instr_prompt,
            *self.response_generator.get_filled_template(template_filling)
        ]
        response = self.response_generator.generate_response(
            current_prompt,
            template_filling.task_instruction,
            template_filling.input_str,
            False,  # no CoT
            None,
            False,
            False,
        )
        logger.debug(f"Agent [bold blue]{self.short_id}[/]: {response.message}")

        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="policy",
            message=response.message,
            agreement=None,
            solution=None,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        self.coordinator.memory.append(memory)
        return response.message, memory, agreements
