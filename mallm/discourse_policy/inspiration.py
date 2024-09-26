from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

from rich.progress import Console  # type: ignore

from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.utils.types import Agreement, Memory, TemplateFilling

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
    from mallm.utils.config import Config

logger = logging.getLogger("mallm")


class InspirationDebate(DiscoursePolicy):
    def draft_proposer_call(
        self,
        draft_proposer: DraftProposer,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        pass

    def panelist_call(
        self,
        agent: Panelist,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        pass

    def discuss(
        self,
        coordinator: Coordinator,
        task_instruction: str,
        input_str: str,
        solution: str,
        config: Config,
        console: Optional[Console] = None,
    ) -> tuple[Optional[str], int, list[Agreement], bool]:
        unique_id = 0
        voting_process_string = ""
        self.memories: list[Memory]
        if console is None:
            console = Console()
        logger.warning("Inspiration Debate cant use different response generators")
        logger.warning(
            "Inspiration Debate only works with voting based decision protocols"
        )
        logger.info(
            """Paradigm: Inspiration
                ┌───┐       ┌───┐       ┌───┐
                │A 3│       │A 1│       │A 2│
                └─│─┘       └─│─┘       └─│─┘
                  │           │           │
                  │───────────│───────────│
                  │           │           │
                ┌─┴─┐       ┌─┴─┐       ┌─┴─┐
                │A 3│       │A 1│       │A 2│
                └───┘       └───┘       └───┘
                """
        )

        while (
            not self.decision or config.skip_decision_making
        ) and self.turn < config.max_turns:
            self.turn += 1
            logger.info("Ongoing. Current turn: " + str(self.turn))

            for agent in coordinator.agents:
                discussion_history = None
                if self.memories:
                    discussion_history = f"Your previous answer was:\n\n {self.memories[0].message}\n\nHere are the answers of the other agents:\n\n{"\n\n".join([f'({memory.persona}) {memory.message}' for memory in self.memories[1:]])}"

                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft="",
                    persona=agent.persona,
                    persona_description=agent.persona_description,
                    agent_memory=discussion_history,
                )

                prompt = [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant taking part in a discussion to solve a task. Your assigned persona is called {agent.persona} and can be described as {agent.persona_description}.",
                    },
                    {
                        "role": "user",
                        "content": f"Your task is: {task_instruction}\n\nThe input you have to work with is: {input_str}.\n\n{discussion_history}\n\nTry to improve your previous answer by critically taking into account the answers from the other participants.",
                    },
                ]

                response = agent.response_generator.generate_response(
                    prompt, task_instruction, input_str, False, None, False, False
                )

                agreement = Agreement(
                    agreement=None,
                    response=response.message,
                    solution=response.solution,
                    agent_id=agent.id,
                    persona=agent.persona,
                    message_id=unique_id,
                )
                self.agreements.append(agreement)

                memory = Memory(
                    message_id=unique_id,
                    turn=self.turn,
                    agent_id=agent.id,
                    persona=agent.persona,
                    contribution="improve",
                    message=response.message,
                    agreement=None,
                    solution=response.solution,
                    memory_ids=[memory.message_id for memory in self.memories],
                    additional_args=dataclasses.asdict(template_filling),
                )
                unique_id += 1
                self.memories.append(memory)

            coordinator.update_memories(self.memories, coordinator.agents)
            self.memories = []
            self.draft, self.decision, self.agreements, voting_process_string = (
                coordinator.decision_protocol.make_decision(
                    self.agreements,
                    self.turn,
                    len(coordinator.agents),
                    task_instruction,
                    input_str,
                )
            )
            self.print_messages(coordinator, input_str, task_instruction)

        self.print_messages(
            coordinator,
            input_str,
            task_instruction,
            False,
            solution,
            voting_process_string,
            console,
        )
        return self.draft, self.turn, self.agreements, self.decision
