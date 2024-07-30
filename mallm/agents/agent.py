from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import TYPE_CHECKING, Optional

import httpx

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator

if TYPE_CHECKING:
    from mallm.agents.moderator import Moderator
    from mallm.coordinator import Coordinator

from mallm.utils.types import Agreement, Memory, TemplateFilling

logger = logging.getLogger("mallm")


class Agent:
    def __init__(
        self,
        llm: Chat,
        client: httpx.Client,
        coordinator: Coordinator,
        response_generator: ResponseGenerator,
        persona: str,
        persona_description: str,
        moderator: Optional[Moderator] = None,
        chain_of_thought: bool = False,
        feedback_only: bool = False,
    ):
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.persona = persona
        self.persona_description = persona_description
        self.coordinator = coordinator
        self.moderator = moderator
        self.llm = llm
        self.response_generator = response_generator
        self.client = client
        self.chain_of_thought = chain_of_thought
        self.feedback_only = feedback_only
        self.memory : dict[str, Memory] = {}
        logger.info(
            f"Creating agent [bold blue]{self.short_id}[/] with personality [bold blue]{self.persona}[/]: {self.persona_description}"
        )

    def improve(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agreements: list[Agreement],
    ) -> tuple[str, Memory, list[Agreement]]:
        logger.debug(f"Agent [bold blue]{self.short_id}[/] is improving the solution.")
        response = self.response_generator.generate_improve(
            template_filling, self.chain_of_thought
        )
        logger.debug(
            f"Agent [bold blue]{self.short_id}[/] {'agreed' if response.agreement else 'disagreed'} with the solution."
        )
        agreements.append(
            Agreement(
                agreement=False if unique_id == 0 else response.agreement,
                response=response.message,
                solution=response.solution,
                agent_id=self.id,
                persona=self.persona,
                message_id=unique_id,
            )
        )

        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="improve",
            message=response.message,
            agreement=agreements[-1].agreement,
            solution=response.solution,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        self.coordinator.update_global_memory(memory)
        return response.message, memory, agreements

    def draft(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agreements: list[Agreement],
        is_moderator: bool = False,
    ) -> tuple[str, Memory, list[Agreement]]:
        logger.debug(f"Agent [bold blue]{self.short_id}[/] is drafting a solution.")
        response = self.response_generator.generate_draft(
            template_filling, self.chain_of_thought
        )
        agreements.append(
            Agreement(
                agreement=None,
                response=response.message,
                solution=response.solution,
                agent_id=self.id,
                persona=self.persona,
                message_id=unique_id,
            )
        )

        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="draft",
            message=response.message,
            agreement=None,
            solution=response.solution,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        self.coordinator.update_global_memory(memory)
        return response.message, memory, agreements

    def feedback(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agreements: list[Agreement],
    ) -> tuple[str, Memory, list[Agreement]]:
        logger.debug(
            f"Agent [bold blue]{self.short_id}[/] provides feedback to a solution."
        )
        response = self.response_generator.generate_feedback(
            template_filling, self.chain_of_thought
        )
        logger.debug(
            f"Agent [bold blue]{self.short_id}[/] {'agreed' if response.agreement else 'disagreed'} with the solution."
        )
        logger.debug(f"Agent [bold blue]{self.short_id}[/]: {response.message}")
        agreements.append(
            Agreement(
                agreement=response.agreement,
                response=response.message,
                solution="",
                agent_id=self.id,
                persona=self.persona,
                message_id=unique_id,
            )
        )

        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="feedback",
            message=response.message,
            agreement=agreements[-1].agreement,
            solution=None,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        self.coordinator.update_global_memory(memory)
        return response.message, memory, agreements

    def update_memory(self, memory: Memory) -> None:
        """
        Updates the memory with another discussion entry.
        """
        self.memory[str(memory.message_id)] = memory

    def get_memories(
        self,
        context_length: Optional[int] = None,
        turn: Optional[int] = None,
        include_this_turn: bool = True,
    ) -> tuple[Optional[list[Memory]], list[int], Optional[str]]:
        """
        Retrieves memory from the agents memory bucket as a Memory
        Returns: Memory
        """
        memories: list[Memory] = []
        memory_ids = []
        current_draft = None

        for key in self.memory.keys():
            memories.extend([self.memory[key]])
        memories = sorted(memories, key=lambda x: x.message_id, reverse=False)
        context_memory = []
        for memory in memories:
            if (
                context_length
                and turn
                and memory.turn >= turn - context_length
                and (turn > memory.turn or include_this_turn)
            ):
                context_memory.append(memory)
                memory_ids.append(int(memory.message_id))
                if memory.contribution == "draft" or (
                    memory.contribution == "improve" and memory.agreement is False
                ):
                    current_draft = memory.solution
            else:
                context_memory.append(memory)
                memory_ids.append(int(memory.message_id))
                if memory.contribution == "draft" or (
                    memory.contribution == "improve" and memory.agreement is False
                ):
                    current_draft = memory.solution

        return context_memory, memory_ids, current_draft

    def get_discussion_history(
        self,
        context_length: Optional[int] = None,
        turn: Optional[int] = None,
        include_this_turn: bool = True,
    ) -> tuple[Optional[list[dict[str, str]]], list[int], Optional[str]]:
        """
        Retrieves memory from the agents memory bucket as a string
        context_length refers to the amount of turns the agent can use as rationale
        Returns: string
        """
        memories, memory_ids, current_draft = self.get_memories(
            context_length=context_length,
            turn=turn,
            include_this_turn=include_this_turn,
        )
        if memories:
            debate_history = []
            for memory in memories:
                if memory.agent_id == self.id:
                    debate_history.append(
                        {"role": "assistant", "content": memory.message}
                    )
                else:
                    debate_history.append(
                        {
                            "role": "user",
                            "content": f"{memory.persona}: {memory.message}",
                        }
                    )
        else:
            debate_history = None
        return debate_history, memory_ids, current_draft
