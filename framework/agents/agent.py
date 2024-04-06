import os
from config import *
import dbm
import json
import ast
import fire
from framework.prompts import agent_prompts
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


class Agent:
    def __init__(self, id, llm, persona, persona_description, coordinator, moderator=None):
        self.id = id
        self.persona = persona
        self.persona_description = persona_description
        self.memory_bucket = memory_bucket_dir + "agent_{}".format(self.id)
        self.coordinator = coordinator
        self.moderator = moderator
        self.llm = llm
        self.init_chains()

    def init_chains(self):
        if "llama" in self.coordinator.llm_tokenizer.__class__.__name__.lower():  # use <<SYS>> and [INST] tokens for llama models
            partial_variables = {"sys_s": "<<SYS>>", "sys_e": "<</SYS>>", "inst_s": "[INST]", "inst_e": "[/INST]"}
        else:
            partial_variables = {"sys_s": "", "sys_e": "", "inst_s": "", "inst_e": ""}

        self.chain_improve = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
            template=agent_prompts.improve(),
            partial_variables=partial_variables))
        self.chain_draft = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
            template=agent_prompts.draft(),
            partial_variables=partial_variables))
        self.chain_feedback = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(
            template=agent_prompts.feedback(),
            partial_variables=partial_variables))

    def improve(self, unique_id, turn, memory_ids, template_filling, extract_all_drafts, agreements):
        res = self.chain_improve.invoke(template_filling)["text"]
        agreements = self.coordinator.agree(res, agreements)
        current_draft = None
        if extract_all_drafts:
            current_draft = self.coordinator.chain_extract_result.invoke(
                {
                    "result": res
                })["text"]
        memory = {
            "unique_id": unique_id,
            "turn": turn,
            "id": self.id,
            "persona": self.persona,
            "contribution": "improve",
            "text": res,
            "agreement": agreements[-1],
            "extracted_draft": current_draft,
            "memory_ids": memory_ids,
            "template_filling": template_filling
        }
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "improve", res, agreements[-1],
                                            None, memory_ids, template_filling)
        return res, memory, agreements

    def draft(self, unique_id, turn, memory_ids, template_filling, extract_all_drafts, agreements, is_moderator=False):
        res = self.chain_draft.invoke(template_filling)["text"]
        agreements = self.coordinator.agree(res, agreements, is_moderator, self_drafted=True)
        current_draft = None
        if extract_all_drafts:
            current_draft = self.coordinator.chain_extract_result.invoke(
                {
                    "result": res
                })["text"]
        if is_moderator:
            agreement = None
        else:
            agreement = agreements[-1]
        memory = {
            "unique_id": unique_id,
            "turn": turn,
            "id": self.id,
            "persona": self.persona,
            "contribution": "draft",
            "text": res,
            "agreement": agreement,
            "extracted_draft": current_draft,
            "memory_ids": memory_ids,
            "template_filling": template_filling
        }
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "draft", res, agreement, None,
                                            memory_ids, template_filling)
        return res, memory, agreements

    def feedback(self, unique_id, turn, memory_ids, template_filling, agreements):
        res = self.chain_feedback.invoke(template_filling)["text"]
        agreements = self.coordinator.agree(res, agreements)
        memory = {
            "unique_id": unique_id,
            "turn": turn,
            "id": self.id,
            "persona": self.persona,
            "contribution": "feedback",
            "text": res,
            "agreement": agreements[-1],
            "extracted_draft": None,
            "memory_ids": memory_ids,
            "template_filling": template_filling
        }
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "feedback", res, agreements[-1],
                                            None, memory_ids, template_filling)
        return res, memory, agreements

    def updateMemory(self, unique_id, turn, agent_id, persona, contribution, text, agreement, extracted_draft,
                     memory_ids, prompt_args):
        '''
        Updates the dbm memory with another discussion entry.
        Returns string
        '''
        if extracted_draft:
            extracted_draft = str(extracted_draft).replace('"', "'")
        with dbm.open(self.memory_bucket, 'c') as db:
            db[
                str(unique_id)] = f'''{{"turn": {turn}, "agent_id": {agent_id}, "persona": "{str(persona).replace('"', "'")}", "prompt_args":{prompt_args}, "contribution": "{contribution}", "memory_ids": {memory_ids}, "text": "{str(text).replace('"', "'")}", "agreement": {agreement}, "extracted_draft": "{str(extracted_draft).replace('"', "'")}"}}'''
        self.saveMemoryToJson()

    def getMemory(self, context_length=None, turn=None, include_this_turn=True, extract_draft=False):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = {}
        memory_ids = []
        current_draft = None
        if os.path.exists(self.memory_bucket + ".dat"):
            with dbm.open(self.memory_bucket, 'r') as db:
                for key in db.keys():
                    memory[key.decode()] = ast.literal_eval(db[key].decode().replace("\n", "\\n").replace("\t", "\\t"))
            memory = sorted(memory.items(), key=lambda x: x, reverse=False)
            context_memory = []
            for m in memory:
                if context_length:
                    if m[1]["turn"] >= turn - context_length:
                        if turn > m[1]["turn"] or include_this_turn:
                            context_memory.append(m)
                            memory_ids.append(int(m[0]))
                            if m[1]["contribution"] == "draft" or (
                                    m[1]["contribution"] == "improve" and "disagree" in m[1]["text"].lower()):
                                current_draft = m[1]["text"]
                else:
                    context_memory.append(m)
                    memory_ids.append(int(m[0]))
                    if m[1]["contribution"] == "draft" or (
                            m[1]["contribution"] == "improve" and "disagree" in m[1]["text"].lower()):
                        current_draft = m[1]["text"]

            context_memory = dict(context_memory)
        else:
            context_memory = None

        if current_draft and extract_draft:
            current_draft = self.coordinator.chain_extract_result.invoke(
                {
                    "result": current_draft
                })["text"]
        return context_memory, memory_ids, current_draft

    def getMemoryString(self, context_length=None, turn=None, personalized=True, include_this_turn=True,
                        extract_draft=False):
        '''
        Retrieves memory from the agents memory bucket as a string
        context_length refers to the amount of turns the agent can use as rationale
        Returns: string
        '''
        memory, memory_ids, current_draft = self.getMemory(context_length=context_length, turn=turn,
                                                           include_this_turn=include_this_turn,
                                                           extract_draft=extract_draft)
        if memory:
            memory_string = ""
            for key in memory:
                if memory[key]["persona"] != self.persona:
                    memory_string = memory_string + f"\n[INST]{memory[key]["persona"]}: {memory[key]["text"]}[/INST]"
                else:
                    memory_string = memory_string + f"\n{memory[key]["persona"]}: {memory[key]["text"]}"
            if personalized:
                memory_string = memory_string.replace(f"{self.persona}:", f"{self.persona} (you):")
        else:
            memory_string = "None"
        return memory_string, memory_ids, current_draft

    def saveMemoryToJson(self, out=None):
        '''
        Converts the memory bucket dbm data to json format
        '''
        if out:
            try:
                with open(out, "w") as f:
                    json.dump(self.getMemory(), f)
            except Exception as e:
                print(f"Failed to save agent memory to {out}: {e}")
        else:
            try:
                with open(self.memory_bucket + ".json", "w") as f:
                    json.dump(self.getMemory(), f)
            except Exception as e:
                print(f"Failed to save agent memory to {self.memory_bucket}: {e}")


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
