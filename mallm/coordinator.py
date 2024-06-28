import dataclasses
import dbm
import json
import logging
import random
import time
import uuid
from collections.abc import Sequence
from datetime import timedelta
from typing import Optional

import httpx

from mallm.agents.agent import Agent
from mallm.agents.moderator import Moderator
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
from mallm.utils.types import Agreement, Memory

logger = logging.getLogger("mallm")


class Coordinator:
    def __init__(
        self,
        model: Chat,
        client: httpx.Client,
        agent_generator: str = "expert",
        use_moderator: bool = False,
        memory_bucket_dir: str = "./mallm/utils/memory_bucket/",
    ):
        self.personas = None
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.panelists: list[Panelist] = []
        self.agents: Sequence[Agent] = []
        self.use_moderator = use_moderator
        self.moderator: Optional[Moderator] = None
        self.memory_bucket_dir = memory_bucket_dir
        self.memory_bucket = self.memory_bucket_dir + "global_" + self.id
        self.decision_protocol: Optional[DecisionProtocol] = None
        self.llm = model
        self.response_generator: ResponseGenerator = SimpleResponseGenerator(self.llm)
        self.client = client
        self.agent_generator = agent_generator

    def init_agents(
        self,
        task_instruction: str,
        input_str: str,
        use_moderator: bool,
        num_agents: int,
        chain_of_thought: bool,
        response_generator: str = "json",
    ) -> None:
        """
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        """
        self.panelists = []
        self.agents = []

        if self.agent_generator not in PERSONA_GENERATORS:
            logger.error(
                f"Invalid persona generator: {self.agent_generator}. Please choose one of: {', '.join(PERSONA_GENERATORS.keys())}"
            )
            raise Exception("Invalid persona generator.")

        if response_generator not in RESPONSE_GENERATORS:
            logger.error(f"No valid response generator for {response_generator}")
            raise Exception(f"No valid response generator for {response_generator}")
        self.response_generator = RESPONSE_GENERATORS[response_generator](self.llm)

        num_agents = 5
        logger.warn(f"Overriding agent generator to generate {num_agents} agents")

        personas = PERSONA_GENERATORS[self.agent_generator](
            llm=self.llm
        ).generate_personas(
            task_description=f"{task_instruction} {input_str}", num_agents=num_agents
        )
        # Add a Neutral Agent with a very Neutral Description
        personas.append(
            {
                "role": "Neutral",
                "description": "Neutral",
            }
        )

        if use_moderator:
            self.moderator = Moderator(
                self.llm, self.client, self, response_generator=self.response_generator
            )
        for persona in personas:
            self.panelists.append(
                Panelist(
                    llm=self.llm,
                    client=self.client,
                    coordinator=self,
                    response_generator=self.response_generator,
                    persona=persona["role"],
                    persona_description=persona["description"],
                    chain_of_thought=chain_of_thought,
                )
            )

        if use_moderator and self.moderator is not None:
            self.agents = [self.moderator, *self.panelists]
        else:
            self.agents = self.panelists

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

    def update_global_memory(self, memory: Memory) -> None:
        """
        Updates the dbm memory with another discussion entry.
        Returns string
        """
        with dbm.open(self.memory_bucket, "c") as db:
            db[str(memory.message_id)] = json.dumps(dataclasses.asdict(memory))
            logger.debug(str(db[str(memory.message_id)]))
        self.save_global_memory_to_json()

    def get_global_memory(self) -> list[Memory]:
        """
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        """
        memory = []
        with dbm.open(self.memory_bucket, "r") as db:
            for key in db.keys():
                json_object = json.loads(db[key].decode())
                memory.append(Memory(**json_object))
        return memory

    def save_global_memory_to_json(self) -> None:
        """
        Converts the memory bucket dbm data to json format
        """
        try:
            with open(self.memory_bucket + ".json", "w") as f:
                json.dump(
                    [dataclasses.asdict(memory) for memory in self.get_global_memory()],
                    f,
                )
        except Exception as e:
            logger.error(f"Failed to save agent memory to {self.memory_bucket}: {e}")
            logger.error(self.get_global_memory())

    def clear_global_memory(self) -> None:
        """
        Clears the memory bucket.
        """
        logger.info(f"Clearing memory bucket {self.memory_bucket}")

        with dbm.open(self.memory_bucket, "n") as _:
            pass

        del self.get_global_memory()[:]

        for agent in self.agents:
            agent.clear_memory()

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

    def setup_personas(
        self, config: Config, input_lines: list[str], context: Optional[list[str]]
    ) -> None:
        self.sample_instruction = config.instruction
        if context:
            self.sample_instruction += "\nHere is some context you need to consider:"
            for c in context:
                self.sample_instruction += "\n" + c
        self.input_str = ""
        for num, input_line in enumerate(input_lines):
            if len(input_lines) > 1:
                self.input_str += str(num + 1) + ") " + input_line + "\n"
            else:
                self.input_str = input_line

        self.init_agents(
            self.sample_instruction,
            self.input_str,
            use_moderator=config.use_moderator,
            num_agents=0,
            chain_of_thought=config.chain_of_thought,
            response_generator=config.response_generator,
        )

    def discuss(self, config: Config) -> tuple[
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

        Returns the final response agreed on, the global memory, agent specific memory, turns needed, last agreements of agents
        """

        mypanelists = random.sample(self.panelists, 3)

        if config.decision_protocol not in DECISION_PROTOCOLS:
            logger.error(f"No valid decision protocol for {config.decision_protocol}")
            raise Exception(
                f"No valid decision protocol for {config.decision_protocol}"
            )
        self.decision_protocol = DECISION_PROTOCOLS[config.decision_protocol](
            mypanelists, config.use_moderator
        )

        start_time = time.perf_counter()

        if config.paradigm not in DISCUSSION_PARADIGMS:
            logger.error(f"No valid discourse policy for paradigm {config.paradigm}")
            raise Exception(f"No valid discourse policy for paradigm {config.paradigm}")
        policy: DiscoursePolicy = DISCUSSION_PARADIGMS[config.paradigm]()

        logger.info(
            f"""Starting discussion with coordinator {self.id}...
-------------
Instruction: {self.sample_instruction}
Input: {self.input_str}
Feedback sentences: {config.feedback_sentences!s}
Maximum turns: {config.max_turns}
Panelists: {[a.persona for a in mypanelists]!s}
Paradigm: {policy.__class__.__name__}
Decision-protocol: {self.decision_protocol.__class__.__name__}
-------------"""
        )

        answer, turn, agreements, decision_success = policy.discuss(
            coordinator=self,
            task_instruction=self.sample_instruction,
            input_str=self.input_str,
            use_moderator=config.use_moderator,
            feedback_sentences=config.feedback_sentences,
            max_turns=config.max_turns,
            force_all_turns=config.force_all_turns,
            context_length=config.context_length,
            include_current_turn_in_memory=config.include_current_turn_in_memory,
            debate_rounds=config.debate_rounds,
        )

        discussion_time = timedelta(
            seconds=time.perf_counter() - start_time
        ).total_seconds()

        global_mem = self.get_global_memory()
        agent_mems = [a.get_memories()[0] for a in self.agents]

        return (
            answer,
            global_mem,
            agent_mems,
            turn,
            agreements,
            discussion_time,
            decision_success,
        )
