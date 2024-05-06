import dbm
import json
import logging
import os
import uuid

import fire
from langchain.chains import LLMChain
from langchain_core.language_models import LLM

from mallm.prompts import agent_prompts
from mallm.prompts.agent_prompts import (
    generate_chat_prompt_improve,
    generate_chat_prompt_feedback,
    generate_chat_prompt_draft,
)
from mallm.prompts.coordinator_prompts import generate_chat_prompt_extract_result

logger = logging.getLogger("mallm")


class Agent:
    def __init__(
        self,
        llm: LLM,
        client,
        persona,
        persona_description,
        coordinator,
        moderator=None,
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
            f'Creating agent {self.short_id} with personality "{self.persona}": "{self.persona_description}"'
        )

    def agree(self, res, agreements, self_drafted=False):
        """
        Determines whether a string given by an agent means an agreement or disagreement.
        Returns bool
        """
        if ("agree" in res.lower() and "disagree" not in res.lower()) and (
            not self == self.moderator
        ):
            agreements.append(
                {"agentId": self.id, "persona": self.persona, "agreement": True}
            )
            logger.debug(f"Agent {self.short_id} agreed")
        elif self_drafted and not self == self.moderator:
            agreements.append(
                {"agentId": self.id, "persona": self.persona, "agreement": True}
            )
            logger.debug(f"Agent {self.short_id} agreed")
        elif not self == self.moderator:
            agreements.append(
                {"agentId": self.id, "persona": self.persona, "agreement": False}
            )
            logger.debug(f"Agent {self.short_id} disagreed")

        if len(agreements) > len(self.coordinator.panelists):
            agreements = agreements[-len(self.coordinator.panelists) :]
        return agreements

    def improve(
        self,
        unique_id,
        turn,
        memory_ids,
        template_filling,
        extract_all_drafts,
        agreements,
    ):
        res = self.llm.invoke(
            generate_chat_prompt_improve(template_filling), client=self.client
        )
        agreements = self.agree(res, agreements)
        current_draft = None
        if extract_all_drafts:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(res), client=self.client
            )
        memory = {
            "messageId": unique_id,
            "turn": turn,
            "agentId": self.id,
            "persona": self.persona,
            "contribution": "improve",
            "text": res,
            "agreement": agreements[-1]["agreement"],
            "extractedDraft": current_draft,
            "memoryIds": memory_ids,
            "additionalArgs": template_filling,
        }
        logger.debug(f"Agent {self.short_id} is improving answer")
        self.coordinator.updateGlobalMemory(
            unique_id,
            turn,
            self.id,
            self.persona,
            "improve",
            res,
            agreements[-1]["agreement"],
            None,
            memory_ids,
            template_filling,
        )
        return res, memory, agreements

    def draft(
        self,
        unique_id,
        turn,
        memory_ids,
        template_filling,
        extract_all_drafts,
        agreements,
        is_moderator=False,
    ):
        res = self.llm.invoke(
            generate_chat_prompt_draft(template_filling), client=self.client
        )
        agreements = self.agree(res, agreements, self_drafted=True)
        current_draft = None
        if extract_all_drafts:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(res), client=self.client
            )
        if is_moderator:
            agreement = None
        else:
            agreement = agreements[-1]
        memory = {
            "messageId": unique_id,
            "turn": turn,
            "agentId": self.id,
            "persona": self.persona,
            "contribution": "draft",
            "text": res,
            "agreement": agreement["agreement"],
            "extractedDraft": current_draft,
            "memoryIds": memory_ids,
            "additionalArgs": template_filling,
        }
        logger.debug(f"Agent {self.short_id} is drafting")
        self.coordinator.updateGlobalMemory(
            unique_id,
            turn,
            self.id,
            self.persona,
            "draft",
            res,
            agreement["agreement"],
            None,
            memory_ids,
            template_filling,
        )
        return res, memory, agreements

    def feedback(self, unique_id, turn, memory_ids, template_filling, agreements):
        res = self.llm.invoke(
            generate_chat_prompt_feedback(template_filling), client=self.client
        )
        agreements = self.agree(res, agreements)
        memory = {
            "messageId": unique_id,
            "turn": turn,
            "agentId": self.id,
            "persona": self.persona,
            "contribution": "feedback",
            "text": res,
            "agreement": agreements[-1]["agreement"],
            "extractedDraft": None,
            "memoryIds": memory_ids,
            "additionalArgs": template_filling,
        }
        logger.debug(f"Agent {self.short_id} provides feedback to another agent")
        self.coordinator.updateGlobalMemory(
            unique_id,
            turn,
            self.id,
            self.persona,
            "feedback",
            res,
            agreements[-1]["agreement"],
            None,
            memory_ids,
            template_filling,
        )
        return res, memory, agreements

    def update_memory(
        self,
        unique_id,
        turn,
        agent_id,
        persona,
        contribution,
        text,
        agreement,
        extracted_draft,
        memory_ids,
        prompt_args,
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
        self.save_memory_to_json()

    def get_memory(
        self,
        context_length=None,
        turn=None,
        include_this_turn=True,
        extract_draft=False,
    ):
        """
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        """
        memory = []
        memory_ids = []
        current_draft = None
        if os.path.exists(self.memory_bucket + ".dat"):
            with dbm.open(self.memory_bucket, "r") as db:
                for key in db.keys():
                    memory.append(
                        json.loads(db[key].decode())
                    )  # TODO: Maybe reverse sort
            # memory = sorted(memory.items(), key=lambda x: x["messageId"], reverse=False)
            context_memory = []
            for m in memory:
                if context_length:
                    if m["turn"] >= turn - context_length:
                        if turn > m["turn"] or include_this_turn:
                            context_memory.append(m)
                            memory_ids.append(int(m["messageId"]))
                            if m["contribution"] == "draft" or (
                                m["contribution"] == "improve"
                            ):
                                current_draft = m["text"]
                else:
                    context_memory.append(m)
                    memory_ids.append(int(m["messageId"]))
                    if m["contribution"] == "draft" or (m["contribution"] == "improve"):
                        current_draft = m["text"]
        else:
            context_memory = None

        if current_draft and extract_draft:
            current_draft = self.llm.invoke(
                generate_chat_prompt_extract_result(current_draft), client=self.client
            )
        return context_memory, memory_ids, current_draft

    def get_memory_string(
        self,
        context_length=None,
        turn=None,
        personalized=True,
        include_this_turn=True,
        extract_draft=False,
    ):
        """
        Retrieves memory from the agents memory bucket as a string
        context_length refers to the amount of turns the agent can use as rationale
        Returns: string
        """
        memory, memory_ids, current_draft = self.get_memory(
            context_length=context_length,
            turn=turn,
            include_this_turn=include_this_turn,
            extract_draft=extract_draft,
        )
        if memory:
            memory_string = ""
            for m in memory:
                memory_string = memory_string + f"""\n{m["persona"]}: {m["text"]}"""
            if personalized:
                memory_string = memory_string.replace(
                    f"""{self.persona}:""", f"""{self.persona} (you):"""
                )
        else:
            memory_string = None
        return memory_string, memory_ids, current_draft

    def save_memory_to_json(self, out=None):
        """
        Converts the memory bucket dbm data to json format
        """
        if out:
            try:
                with open(out, "w") as f:
                    json.dump(self.get_memory(), f)
            except Exception as e:
                print(f"Failed to save agent memory to {out}: {e}")
        else:
            try:
                with open(self.memory_bucket + ".json", "w") as f:
                    json.dump(self.get_memory(), f)
            except Exception as e:
                print(f"Failed to save agent memory to {self.memory_bucket}: {e}")


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
