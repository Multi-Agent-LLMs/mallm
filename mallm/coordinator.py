import dataclasses
import dbm
import json
import logging
import os
import time
import uuid
from datetime import timedelta
from typing import Optional, Sequence, Type

import httpx

from mallm.agents.agent import Agent
from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.decision_making.decision_protocol import DecisionProtocol
from mallm.decision_making.majority import MajorityConsensus
from mallm.decision_making.voting import Voting
from mallm.discourse_policy.DiscourceDebate import DiscourseDebate
from mallm.discourse_policy.DiscourceMemory import DiscourseMemory
from mallm.discourse_policy.DiscourceRelay import DiscourseRelay
from mallm.discourse_policy.DiscourceReport import DiscourseReport
from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from mallm.models.Chat import Chat
from mallm.models.personas.ExpertGenerator import ExpertGenerator
from mallm.prompts.coordinator_prompts import generate_chat_prompt_extract_result
from mallm.utils.types import Agreement, Memory

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

logger = logging.getLogger("mallm")

DECISION_PROTOCOLS: dict[str, Type[DecisionProtocol]] = {
    "majority_consensus": MajorityConsensus,
    "voting": Voting,
}

PROTOCOLS: dict[str, Type[DiscoursePolicy]] = {
    "memory": DiscourseMemory,
    "report": DiscourseReport,
    "relay": DiscourseRelay,
    "debate": DiscourseDebate,
}


class Coordinator:
    def __init__(
        self,
        model: Chat,
        client: httpx.Client,
        agent_generator: Optional[ExpertGenerator] = None,
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
        self.decision_making: Optional[DecisionProtocol] = None
        self.llm = model
        self.client = client
        self.agent_generator = agent_generator

    def init_agents(
        self, task_instruction: str, input_str: str, use_moderator: bool
    ) -> None:
        """
        Instantiates the agents by
        1) identify helpful personas
        2) create agents with the personas
        """
        self.panelists = []
        self.agents = []

        if self.agent_generator is None:
            logger.error("No persona generator provided.")
            raise Exception("No persona generator provided.")

        personas = self.agent_generator.generate_personas(
            f"{task_instruction} {input_str}", 3
        )

        if use_moderator:
            self.moderator = Moderator(self.llm, self.client, self)
        for persona in personas:
            self.panelists.append(
                Panelist(
                    self.llm, self.client, self, persona["role"], persona["description"]
                )
            )

        if use_moderator and self.moderator is not None:
            self.agents = [self.moderator, *self.panelists]
        else:
            self.agents = self.panelists

    def get_agents(self) -> list[dict[str, str]]:
        agent_dicts = []
        for a in self.agents:
            agent_dicts.append(
                {
                    "agentId": a.id,
                    "model": "placeholder",  # TODO: automatically detect model name
                    "persona": a.persona,
                    "personaDescription": a.persona_description,
                }
            )
        return agent_dicts

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

    def update_memories(
        self, memories: list[Memory], agents_to_update: Sequence[Agent]
    ) -> None:
        """
        Updates the memories of all declared agents.
        """
        for memory in memories:
            for agent in agents_to_update:
                agent.update_memory(memory)

    def discuss(
        self,
        task_instruction: str,
        input_str: str,
        context: list[str],
        use_moderator: bool,
        feedback_sentences: tuple[int, int],
        paradigm: str,
        decision_protocol: str,
        max_turns: int,
        context_length: int,
        include_current_turn_in_memory: bool,
        extract_all_drafts: bool,
        debate_rounds: Optional[int],
    ) -> tuple[
        str, list[Memory], list[Optional[list[Memory]]], int, list[Agreement], float
    ]:
        """
        The routine responsible for the discussion between agents to solve a task.

        The routine is organized as follows:
        1) Create agents with personas
        2) Discuss the problem based on the given paradigm (iteratively check for agreement between agents)
        3) After max turns or agreement reached: return the final result to the task sample

        Returns the final response agreed on, the global memory, agent specific memory, turns needed, last agreements of agents
        """
        if context and isinstance(context, list):
            for c in context:
                task_instruction += "\n" + c

        self.init_agents(task_instruction, input_str, use_moderator=use_moderator)

        if decision_protocol not in DECISION_PROTOCOLS:
            logger.error(f"No valid decision protocol for {decision_protocol}")
            raise Exception(f"No valid decision protocol for {decision_protocol}")

        self.decision_making = DECISION_PROTOCOLS[decision_protocol](self.panelists)

        start_time = time.perf_counter()

        if paradigm not in PROTOCOLS:
            logger.error(f"No valid discourse policy for paradigm {paradigm}")
            raise Exception(f"No valid discourse policy for paradigm {paradigm}")
        policy: DiscoursePolicy = PROTOCOLS[paradigm]()

        logger.info(
            f"""Starting discussion with coordinator {self.id}...
-------------
Instruction: {task_instruction}
Input: {input_str}
Feedback sentences: {feedback_sentences!s}
Maximum turns: {max_turns}
Agents: {[a.persona for a in self.agents]!s}
Paradigm: {policy.__class__.__name__}
Decision-making: {self.decision_making.__class__.__name__}
-------------"""
        )

        current_draft, turn, agreements = policy.discuss(
            self,
            task_instruction,
            input_str,
            use_moderator,
            feedback_sentences,
            max_turns,
            context_length,
            include_current_turn_in_memory,
            extract_all_drafts,
        )

        discussion_time = timedelta(
            seconds=time.perf_counter() - start_time
        ).total_seconds()

        global_mem = self.get_global_memory()
        agent_mems = []
        for a in self.agents:
            agent_mems.append(a.get_memories()[0])
        if turn >= max_turns:  # if no agreement was reached
            current_draft = "No agreement was reached."
        else:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(input_str, current_draft),
                client=self.client,
            )

        return current_draft, global_mem, agent_mems, turn, agreements, discussion_time
