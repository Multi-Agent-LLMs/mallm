from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from rich.panel import Panel
from rich.progress import Console  # type: ignore
from rich.text import Text

from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.utils.types import Agreement, Memory, TemplateFilling, VotingResults

if TYPE_CHECKING:
    from mallm.coordinator import Coordinator
    from mallm.utils.config import Config
logger = logging.getLogger("mallm")


class DiscoursePolicy(ABC):
    def __init__(self, paradigm_str: str = "") -> None:
        self.paradigm_str = paradigm_str
        self.decision = False
        self.turn = 0
        self.unique_id = 0
        self.memories: list[Memory] = []
        self.draft = ""
        self.agreements: list[Agreement] = []

    def discuss(
        self,
        coordinator: Coordinator,
        task_instruction: str,
        input_str: str,
        config: Config,
        console: Optional[Console] = None,
    ) -> tuple[Optional[str], int, list[Agreement], bool, Optional[VotingResults]]:
        logger.info(self.paradigm_str)
        voting_process_string = ""
        additional_voting_results: Optional[VotingResults] = None
        if console is None:
            console = Console()
        while (
            not self.decision or config.skip_decision_making
        ) and self.turn < config.max_turns:
            self.turn += 1
            logger.info(f"Ongoing. Current turn: {self.turn}")

            for i, agent in enumerate(coordinator.agents):
                discussion_history, memory_ids, current_draft = (
                    agent.get_discussion_history(
                        context_length=config.visible_turns_in_memory,
                        turn=self.turn,
                    )
                )
                if self.turn == 1 and config.all_agents_generate_first_draft:
                    current_draft = None
                    discussion_history = None

                template_filling = TemplateFilling(
                    task_instruction=task_instruction,
                    input_str=input_str,
                    current_draft=current_draft,
                    persona=agent.persona,
                    persona_description=agent.persona_description,
                    agent_memory=discussion_history,
                )

                if isinstance(agent, DraftProposer):
                    template_filling.feedback_sentences = None
                    self.draft_proposer_call(
                        draft_proposer=agent,
                        template_filling=template_filling,
                        memory_ids=memory_ids,
                        agent_index=i,
                        coordinator=coordinator,
                    )
                elif isinstance(agent, Panelist):
                    self.panelist_call(
                        agent_index=i,
                        template_filling=template_filling,
                        coordinator=coordinator,
                        memory_ids=memory_ids,
                        agent=agent,
                    )
                else:
                    logger.error("Agent type not recognized.")
                    raise Exception("Agent type not recognized.")
                self.unique_id += 1
                self.memories = []

                if coordinator.decision_protocol is None:
                    logger.error("No decision protocol module found.")
                    raise Exception("No decision protocol module found.")

                (
                    self.draft,
                    self.decision,
                    self.agreements,
                    voting_process_string,
                    additional_voting_results,
                ) = coordinator.decision_protocol.make_decision(
                    self.agreements, self.turn, i, task_instruction, input_str, config
                )

                if self.decision:
                    break

            self.print_messages(coordinator, input_str, task_instruction)

        self.print_messages(
            coordinator,
            input_str,
            task_instruction,
            False,
            voting_process_string,
            console,
        )
        return (
            self.draft,
            self.turn,
            self.agreements,
            self.decision,
            additional_voting_results,
        )

    def print_messages(
        self,
        coordinator: Coordinator,
        input_str: str,
        task_instruction: str,
        only_current_turn: bool = True,
        voting_process_string: str = "",
        console: Optional[Console] = None,
    ) -> None:
        if console is None:
            console = Console()
        global_memories = [
            memory
            for memory in coordinator.memory
            if memory.turn == self.turn or not only_current_turn
        ]
        max_width = min(console.width, 100)
        discussion_text = Text(
            f"Task instruction: {task_instruction}\n\nInput: {input_str}\n-----------\n"
            + "\n-----------\n".join(
                [
                    f"Agent ({m.persona})({'agreed' if m.agreement else 'disagreed'}): {m.message}"
                    for m in global_memories
                ]
            )
            + f"\n-----------\nDecision Success: {self.decision} \n\nAccepted solution: {self.draft}"
            + (f"\n\n{voting_process_string}" if voting_process_string else "")
        )
        discussion_text.highlight_regex(r"Agent .*\):", style="bold green")
        discussion_text.highlight_regex(r"Task instruction:", style="bold green")
        discussion_text.highlight_regex(r"Input:", style="bold green")
        discussion_text.highlight_regex(r"Decision Success:", style="bold green")
        discussion_text.highlight_regex(r"Accepted solution:", style="bold green")
        discussion_text.highlight_regex(r"Voting with alteration:", style="bold green")
        discussion_text.highlight_regex(r".* final answer:", style="bold green")
        discussion_text.highlight_regex(r"Facts:", style="bold green")
        discussion_text.highlight_regex(r"####.*", style="bold green")
        for panelist in coordinator.panelists:
            discussion_text.highlight_regex(panelist.persona, style="bold blue")
        panel = Panel(
            discussion_text,
            title=(
                f"Discussion Turn {global_memories[0].turn}"
                if only_current_turn
                else "Discussion"
            ),
            subtitle=f"Decision: {self.decision}",
            expand=False,
            width=max_width,
        )
        console.print(panel)

    @abstractmethod
    def draft_proposer_call(
        self,
        draft_proposer: DraftProposer,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        pass

    @abstractmethod
    def panelist_call(
        self,
        agent: Panelist,
        coordinator: Coordinator,
        agent_index: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
    ) -> None:
        pass
