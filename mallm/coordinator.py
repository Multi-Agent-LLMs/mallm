import dbm
import json
import logging
import os
import time
import uuid
from datetime import timedelta
from typing import Optional, Sequence, Type

import fire
import transformers
from openai import OpenAI

from mallm.agents.agent import Agent
from mallm.agents.moderator import Moderator
from mallm.agents.panelist import Panelist
from mallm.decision_making.DecisionProtocol import DecisionProtocol
from mallm.decision_making.MajorityConsensus import MajorityConsensus
from mallm.decision_making.Voting import Voting
from mallm.discourse_policy.DiscourceDebate import DiscourseDebate
from mallm.discourse_policy.DiscourceMemory import DiscourseMemory
from mallm.discourse_policy.DiscourceRelay import DiscourseRelay
from mallm.discourse_policy.DiscourceReport import DiscourseReport
from mallm.discourse_policy.DiscoursePolicy import DiscoursePolicy
from mallm.models.HFTGIChat import HFTGIChat
from mallm.models.personas.PersonaGenerator import PersonaGenerator
from mallm.prompts.coordinator_prompts import generate_chat_prompt_extract_result
from mallm.utils.types.Agreement import Agreement

transformers.logging.set_verbosity_error()
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

logger = logging.getLogger("mallm")

decision_protocols: dict[str, Type[DecisionProtocol]] = {
    "majority_consensus": MajorityConsensus,
    "voting": Voting,
}

protocols: dict[str, Type[DiscoursePolicy]] = {
    "memory": DiscourseMemory,
    "report": DiscourseReport,
    "relay": DiscourseRelay,
    "debate": DiscourseDebate,
}


class Coordinator:
    def __init__(
        self,
        model: HFTGIChat,
        client: OpenAI,
        agent_generator: Optional[PersonaGenerator] = None,
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

    def init_agents(self, task_instruction: str, input_str: str, use_moderator: bool):
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
            self.agents = [agent for agent in [self.moderator] + self.panelists]
        else:
            self.agents = self.panelists

    def get_agents(self):
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

    def update_global_memory(
        self,
        unique_id: int,
        turn: int,
        agent_id: str,
        persona: str,
        contribution: str,
        text: str,
        agreement: bool,
        extracted_draft: str,
        memory_ids: list[int],
        prompt_args: dict,
    ):
        """
        Updates the dbm memory with another discussion entry.
        Returns string
        """

        data_dict = {
            "messageId": unique_id,
            "turn": turn,
            "agentId": agent_id,
            "persona": str(persona).replace('"', "'"),
            "additionalArgs": prompt_args,
            "contribution": contribution,
            "memoryIds": memory_ids,
            "text": str(text).replace('"', "'"),
            "agreement": agreement,
            "extractedDraft": str(extracted_draft).replace('"', "'"),
        }

        with dbm.open(self.memory_bucket, "c") as db:
            db[str(unique_id)] = json.dumps(data_dict)
            logger.debug(str(db[str(unique_id)]))
        self.save_global_memory_to_json()

    def get_global_memory(self):
        """
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        """
        memory = []
        with dbm.open(self.memory_bucket, "r") as db:
            for key in db.keys():
                memory.append(json.loads(db[key].decode()))
        return memory

    def save_global_memory_to_json(self):
        """
        Converts the memory bucket dbm data to json format
        """
        try:
            with open(self.memory_bucket + ".json", "w") as f:
                json.dump(self.get_global_memory(), f)
        except Exception as e:
            logger.error(f"Failed to save agent memory to {self.memory_bucket}: {e}")
            logger.error(self.get_global_memory())

    def update_memories(self, memories: list[dict], agents_to_update: list[Agent]):
        """
        Updates the memories of all declared agents.
        """
        for c in memories:
            for a in agents_to_update:
                a.update_memory(
                    c["messageId"],
                    c["turn"],
                    c["agentId"],
                    c["persona"],
                    c["contribution"],
                    c["text"],
                    c["agreement"],
                    c["extractedDraft"],
                    c["memoryIds"],
                    c["additionalArgs"],
                )
        return []

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
    ) -> tuple[str, list, list, int, list[Agreement], float]:
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

        if decision_protocol not in decision_protocols:
            logger.error(f"No valid decision protocol for {decision_protocol}")
            raise Exception(f"No valid decision protocol for {decision_protocol}")

        self.decision_making = decision_protocols[decision_protocol](self.panelists)

        start_time = time.perf_counter()

        if paradigm not in protocols:
            logger.error(f"No valid discourse policy for paradigm {paradigm}")
            raise Exception(f"No valid discourse policy for paradigm {paradigm}")
        policy: DiscoursePolicy = protocols[paradigm]()

        logger.info(
            f"""Starting discussion with coordinator {self.id}...
-------------
Instruction: {task_instruction}
Input: {input_str}
Feedback sentences: {str(feedback_sentences)}
Maximum turns: {max_turns}
Agents: {str([a.persona for a in self.agents])}
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
            agent_mems.append(a.get_memory()[0])
        if turn >= max_turns:  # if no agreement was reached
            current_draft = "No agreement was reached."
        else:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(input_str, current_draft),
                client=self.client,
            )

        return current_draft, global_mem, agent_mems, turn, agreements, discussion_time


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
