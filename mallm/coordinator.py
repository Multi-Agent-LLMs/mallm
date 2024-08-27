import logging
import time
import uuid
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Optional

import httpx
from rich.progress import Console  # type: ignore

from mallm.agents.agent import Agent
from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.models.discussion.SimpleResponseGenerator import SimpleResponseGenerator
from mallm.utils.config import Config
from mallm.utils.dicts import (
    DECISION_PROTOCOLS,
    DISCUSSION_PARADIGMS,
    PERSONA_GENERATORS,
    RESPONSE_GENERATORS,
)
from mallm.utils.types import Agreement, InputExample, Memory

logger = logging.getLogger("mallm")


class Coordinator:
    def __init__(
        self,
        model: Chat,
        client: httpx.Client,
        agent_generator: str = "expert",
        num_neutral_agents: int = 0,
        console: Optional[Console] = None,
    ):
        self.personas = None
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.panelists: list[Panelist] = []
        self.agents: list[Agent] = []
        self.num_neutral_agents = num_neutral_agents
        self.draft_proposers: list[DraftProposer] = []
        self.decision_protocol: Optional[DecisionProtocol] = None
        self.llm = model
        self.response_generator: ResponseGenerator = SimpleResponseGenerator(self.llm)
        self.client = client
        self.agent_generator = agent_generator
        self.memory: list[Memory] = []
        self.console = console or Console()

    def init_agents(
        self,
        task_instruction: str,
        input_str: str,
        num_neutral_agents: int,
        num_agents: int,
        chain_of_thought: bool,
        all_agents_drafting: bool,
        sample: InputExample,
    ) -> None:
        """
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        """
        logger.debug(f"Coordinator {self.id} creates {num_agents} agents ({self.agent_generator})...")

        num_agents -= num_neutral_agents

        if self.agent_generator not in PERSONA_GENERATORS:
            logger.error(
                f"Invalid persona generator: {self.agent_generator}. Please choose one of: {', '.join(PERSONA_GENERATORS.keys())}"
            )
            raise Exception("Invalid persona generator.")

        personas = PERSONA_GENERATORS[self.agent_generator](
            llm=self.llm
        ).generate_personas(
            task_description=f"{task_instruction} {input_str}",
            num_agents=num_agents,
            sample=sample,
        )
        logger.debug(f"Created {len(personas)} personas: \n" + str(personas))

        for n in range(num_neutral_agents):
            draft_proposer = DraftProposer(
                    self.llm, self.client, self, response_generator=self.response_generator, persona=f"Moderator {n + 1}" if num_neutral_agents > 1 else "Moderator"
                )
            self.draft_proposers.append(
                draft_proposer
            )
            self.agents.append(draft_proposer)

        for persona in personas:
            panelist = Panelist(
                    llm=self.llm,
                    client=self.client,
                    coordinator=self,
                    response_generator=self.response_generator,
                    persona=persona["role"],
                    persona_description=persona["description"],
                    chain_of_thought=chain_of_thought,
                    drafting_agent=all_agents_drafting,
                )
            self.panelists.append(
                panelist
            )
            self.agents.append(panelist)

        if len(self.agents) == 1:
            logger.warning(
                "Created only 1 agent. The discussion will be replaced by a self-improvement mechanism."
            )

    def get_agents(self) -> list[dict[str, str]]:
        return [
            {
                "agentId": a.id,
                "model": a.llm.model,
                "persona": a.persona,
                "personaDescription": a.persona_description,
            }
            for a in self.agents
        ]

    @staticmethod
    def update_memories(
        memories: list[Memory], agents_to_update: Sequence[Agent]
    ) -> None:
        """
        Updates the memories of all declared agents.
        """
        for memory in memories:
            for agent in agents_to_update:
                agent.update_memory(memory)

    def discuss(
        self,
        config: Config,
        sample: InputExample,
    ) -> tuple[
        Optional[str],
        list[Memory],
        list[Optional[list[Memory]]],
        int,
        list[Agreement],
        float,
        bool,
    ]:
        """
        The routine responsible for the discussion between agents to solve a task.

        The routine is organized as follows:
        1) Create agents with personas
        2) Discuss the problem based on the given paradigm (iteratively check for agreement between agents)
        3) After max turns or agreement reached: return the final result to the task sample

        Returns final response, global memory, agent specific memory, turns needed, last agreements of agents, discussion time in seconds, boolean if agreement was reached
        """
        sample_instruction = config.task_instruction_prompt
        if sample.context:
            sample_instruction += "\nContext:"
            for c in sample.context:
                sample_instruction += "\n" + c
        input_str = ""
        for num, input_line in enumerate(sample.inputs):
            if len(sample.inputs) > 1:
                input_str += str(num + 1) + ") " + input_line + "\n"
            else:
                input_str = input_line

        if config.response_generator not in RESPONSE_GENERATORS:
            logger.error(f"No valid response generator for {config.response_generator}")
            raise Exception(
                f"No valid response generator for {config.response_generator}"
            )
        self.response_generator = RESPONSE_GENERATORS[config.response_generator](
            self.llm
        )

        self.init_agents(
            sample_instruction,
            input_str,
            num_neutral_agents=config.num_neutral_agents,
            num_agents=config.num_agents,
            chain_of_thought=config.use_chain_of_thought,
            all_agents_drafting=config.all_agents_drafting,
            sample=sample,
        )

        if config.decision_protocol not in DECISION_PROTOCOLS:
            logger.error(f"No valid decision protocol for {config.decision_protocol}")
            raise Exception(
                f"No valid decision protocol for {config.decision_protocol}"
            )
        self.decision_protocol = DECISION_PROTOCOLS[config.decision_protocol](
            self.panelists, config.num_neutral_agents
        )

        start_time = time.perf_counter()

        if config.discussion_paradigm not in DISCUSSION_PARADIGMS:
            logger.error(f"No valid discourse policy for paradigm {config.discussion_paradigm}")
            raise Exception(f"No valid discourse policy for paradigm {config.discussion_paradigm}")
        policy: DiscoursePolicy = DISCUSSION_PARADIGMS[config.discussion_paradigm]()

        logger.info(
            f"""Starting discussion with coordinator {self.id}...
-------------
[bold blue]Instruction:[/] {sample_instruction}
[bold blue]Input:[/] {input_str}
[bold blue]Maximum turns:[/] {config.max_turns}
[bold blue]Agents:[/] {[a.persona for a in self.agents]!s}
[bold blue]Paradigm:[/] {policy.__class__.__name__}
[bold blue]Decision-protocol:[/] {self.decision_protocol.__class__.__name__}
-------------"""
        )

        answer, turn, agreements, decision_success = policy.discuss(
            coordinator=self,
            task_instruction=sample_instruction,
            input_str=input_str,
            num_neutral_agents=config.num_neutral_agents,
            max_turns=config.max_turns,
            force_all_turns=config.skip_decision_making,
            context_length=config.visible_turns_in_memory,
            debate_rounds=config.debate_rounds,
            console=self.console,
        )

        discussion_time = timedelta(
            seconds=time.perf_counter() - start_time
        ).total_seconds()

        self.console.save_html(str(Path(config.output_json_file_path).with_suffix(".html")), clear=False)

        return (
            answer,
            self.memory,
            [a.get_memories()[0] for a in self.agents],
            turn,
            agreements,
            discussion_time,
            decision_success,
        )
