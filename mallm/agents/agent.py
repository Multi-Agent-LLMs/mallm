from __future__ import annotations

import dataclasses
import dbm
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from mallm.models.Chat import Chat

if TYPE_CHECKING:
    from mallm.agents.moderator import Moderator
    from mallm.coordinator import Coordinator
from mallm.prompts.agent_prompts import (
    generate_chat_prompt_draft,
    generate_chat_prompt_feedback,
    generate_chat_prompt_improve,
)
from mallm.prompts.coordinator_prompts import generate_chat_prompt_extract_result
from mallm.utils.types import Agreement, Memory, TemplateFilling

logger = logging.getLogger("mallm")


class Agent:
    def __init__(
        self,
        llm: Chat,
        client: httpx.Client,
        coordinator: Coordinator,
        persona: str,
        persona_description: str,
        moderator: Optional[Moderator] = None,
    ):
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.persona = persona
        self.persona_description = persona_description
        self.memory_bucket = coordinator.memory_bucket_dir + "agent_{}".format(self.id)
        self.coordinator = coordinator
        self.moderator = moderator
        self.llm = llm
        self.client = client
        logger.info(
            f"Creating agent {self.short_id} with personality {self.persona}: {self.persona_description}"
        )

    def agree(
        self, res: str, agreements: list[Agreement], self_drafted: bool = False
    ) -> list[Agreement]:
        """
        Determines whether a string given by an agent means an agreement or disagreement.
        Returns a list of Agreements
        """
        if (
            "agree" in res.lower()
            and "disagree" not in res.lower()
            and not self_drafted
        ):
            agreements.append(
                Agreement(
                    agreement=True, agent_id=self.id, persona=self.persona, response=res
                )
            )
            logger.debug(f"Agent {self.short_id} agreed")
        else:
            agreements.append(
                Agreement(
                    agreement=False,
                    agent_id=self.id,
                    persona=self.persona,
                    response=res,
                )
            )
            logger.debug(f"Agent {self.short_id} disagreed")

        # Only keep the most recent agreements
        if len(agreements) > len(self.coordinator.agents):
            agreements = agreements[-len(self.coordinator.agents) :]
        return agreements

    def improve(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        extract_all_drafts: bool,
        agreements: list[Agreement],
        chain_of_thought: bool = True,
    ) -> tuple[str, Memory, list[Agreement]]:
        res = self.llm.invoke(
            generate_chat_prompt_improve(
                template_filling, chain_of_thought=chain_of_thought
            ),
            client=self.client,
        )
        agreements = self.agree(res, agreements)
        current_draft = None
        # new drafts are only proposed upon disagreement, thus we do not want to overwrite unnecessarily
        if extract_all_drafts and not agreements[-1].agreement:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(res),
                client=self.client,
            )
        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="improve",
            text=res,
            agreement=agreements[-1].agreement,
            extracted_draft=current_draft,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        logger.debug(f"Agent {self.short_id} is improving answer")
        self.coordinator.update_global_memory(memory)
        return res, memory, agreements

    def draft(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        extract_all_drafts: bool,
        agreements: list[Agreement],
        is_moderator: bool = False,
        chain_of_thought: bool = True,
    ) -> tuple[str, Memory, list[Agreement]]:
        res = self.llm.invoke(
            generate_chat_prompt_draft(
                template_filling, chain_of_thought=chain_of_thought
            ),
            client=self.client,
        )
        agreements = self.agree(res, agreements, self_drafted=True)
        current_draft = None
        if extract_all_drafts:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(res),
                client=self.client,
            )
        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="draft",
            text=res,
            agreement=agreements[-1].agreement,
            extracted_draft=current_draft,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        logger.debug(f"Agent {self.short_id} is drafting")
        self.coordinator.update_global_memory(memory)
        return res, memory, agreements

    def feedback(
        self,
        unique_id: int,
        turn: int,
        memory_ids: list[int],
        template_filling: TemplateFilling,
        agreements: list[Agreement],
        chain_of_thought: bool = True,
    ) -> tuple[str, Memory, list[Agreement]]:
        res = self.llm.invoke(
            generate_chat_prompt_feedback(
                template_filling, chain_of_thought=chain_of_thought
            ),
            client=self.client,
        )
        agreements = self.agree(res, agreements)
        memory = Memory(
            message_id=unique_id,
            turn=turn,
            agent_id=self.id,
            persona=self.persona,
            contribution="feedback",
            text=res,
            agreement=agreements[-1].agreement,
            extracted_draft=None,
            memory_ids=memory_ids,
            additional_args=dataclasses.asdict(template_filling),
        )
        logger.debug(f"Agent {self.short_id} provides feedback to another agent")
        self.coordinator.update_global_memory(memory)
        return res, memory, agreements

    def update_memory(self, memory: Memory) -> None:
        """
        Updates the dbm memory with another discussion entry.
        """
        with dbm.open(self.memory_bucket, "c") as db:
            db[str(memory.message_id)] = json.dumps(dataclasses.asdict(memory))
        self.save_memory_to_json()

    def get_memories(
        self,
        context_length: Optional[int] = None,
        turn: Optional[int] = None,
        include_this_turn: bool = True,
        extract_draft: bool = False,
    ) -> tuple[Optional[list[Memory]], list[int], Optional[str], Optional[str]]:
        """
        Retrieves memory from the agents memory bucket as a Memory
        Returns: Memory
        """
        memories: list[Memory] = []
        memory_ids = []
        current_draft = None

        try:
            with dbm.open(self.memory_bucket, "r") as db:
                for key in db.keys():
                    json_object = json.loads(db[key].decode())
                    memories.append(Memory(**json_object))
            memories = sorted(memories, key=lambda x: x.message_id, reverse=False)
            context_memory = []
            extracted = False
            for memory in memories:
                if context_length:
                    if turn and memory.turn >= turn - context_length:
                        if turn > memory.turn or include_this_turn:
                            context_memory.append(memory)
                            memory_ids.append(int(memory.message_id))
                            if memory.contribution == "draft" or (
                                memory.contribution == "improve"
                                and memory.agreement == False
                            ):
                                if memory.extracted_draft:
                                    current_draft = memory.extracted_draft
                                    extracted = True
                                else:
                                    current_draft = memory.text
                                    extracted = False
                else:
                    context_memory.append(memory)
                    memory_ids.append(int(memory.message_id))
                    if memory.contribution == "draft" or (
                        memory.contribution == "improve" and memory.agreement == False
                    ):
                        if memory.extracted_draft:
                            current_draft = memory.extracted_draft
                            extracted = True
                        else:
                            current_draft = memory.text
                            extracted = False
        except dbm.error:
            context_memory = None

        full_draft = current_draft
        if (
            current_draft != "" and extract_draft and not extracted
        ):  # if not extracted already
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(current_draft),
                client=self.client,
            )
        return context_memory, memory_ids, current_draft, full_draft

    def get_discussion_history(
        self,
        context_length: Optional[int] = None,
        turn: Optional[int] = None,
        include_this_turn: bool = True,
        extract_draft: bool = False,
    ) -> tuple[Optional[list[dict[str, str]]], list[int], Optional[str], Optional[str]]:
        """
        Retrieves memory from the agents memory bucket as a string
        context_length refers to the amount of turns the agent can use as rationale
        Returns: string
        """
        memories, memory_ids, current_draft, full_draft = self.get_memories(
            context_length=context_length,
            turn=turn,
            include_this_turn=include_this_turn,
            extract_draft=extract_draft,
        )
        if memories:
            debate_history = []
            for memory in memories:
                if memory.agent_id == self.id:
                    debate_history.append({"role": "assistant", "content": memory.text})
                else:
                    debate_history.append(
                        {
                            "role": "user",
                            "content": f"{memory.persona}: {memory.text}",
                        }
                    )
        else:
            debate_history = None
        return debate_history, memory_ids, current_draft, full_draft

    def save_memory_to_json(self, out: Optional[str] = None) -> None:
        """
        Converts the memory bucket dbm data to json format
        """
        save_path = out if out else self.memory_bucket + ".json"
        try:
            memories, memory_ids, current_draft, full_draft = self.get_memories()
            if memories:
                with open(save_path, "w") as f:
                    json.dump(
                        (
                            [dataclasses.asdict(memory) for memory in memories],
                            memory_ids,
                            current_draft,
                        ),
                        f,
                    )
        except Exception as e:
            logger.error(f"Failed to save agent memory to {save_path}: {e}")
