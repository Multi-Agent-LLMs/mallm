import os
from mallm.config import *
import dbm
import json
import ast
import fire
from mallm.prompts import agent_prompts
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import uuid


class Agent:
    def __init__(self, llm, llm_tokenizer, persona, persona_description, coordinator, moderator=None):
        self.id = str(uuid.uuid4())
        self.persona = persona
        self.persona_description = persona_description
        self.memory_bucket = memory_bucket_dir + "agent_{}".format(self.id)
        self.coordinator = coordinator
        self.moderator = moderator
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.init_chains()

    def init_chains(self):
        if "llama" in self.llm_tokenizer.__class__.__name__.lower():  # use <<SYS>> and [INST] tokens for llama models
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

    def agree(self, res, agreements, self_drafted=False):
        '''
        Determines whether a string given by an agent means an agreement or disagreement.
        Returns bool
        '''
        if ("agree" in res.lower() and "disagree" not in res.lower()) and (not self == self.moderator):
            agreements.append({ "agentId": self.id, "persona": self.persona, "agreement": True })
        elif self_drafted and not self == self.moderator:
            agreements.append({ "agentId": self.id, "persona": self.persona, "agreement": True })
        elif not self == self.moderator:
            agreements.append({ "agentId": self.id, "persona": self.persona, "agreement": False })

        if len(agreements) > len(self.coordinator.panelists):
            agreements = agreements[-len(self.coordinator.panelists):]
        return agreements
    
    def improve(self, unique_id, turn, memory_ids, template_filling, extract_all_drafts, agreements):
        res = self.chain_improve.invoke(template_filling)["text"]
        agreements = self.agree(res, agreements)
        current_draft = None
        if extract_all_drafts:
            current_draft = self.coordinator.chain_extract_result.invoke(
                {
                    "result": res
                })["text"]
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
            "additionalArgs": template_filling
        }
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "improve", res, agreements[-1]["agreement"],
                                            None, memory_ids, template_filling)
        return res, memory, agreements

    def draft(self, unique_id, turn, memory_ids, template_filling, extract_all_drafts, agreements, is_moderator=False):
        res = self.chain_draft.invoke(template_filling)["text"]
        agreements = self.agree(res, agreements, self_drafted=True)
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
            "messageId": unique_id,
            "turn": turn,
            "agentId": self.id,
            "persona": self.persona,
            "contribution": "draft",
            "text": res,
            "agreement": agreement["agreement"],
            "extractedDraft": current_draft,
            "memoryIds": memory_ids,
            "additionalArgs": template_filling
        }
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "draft", res, agreement["agreement"], None,
                                            memory_ids, template_filling)
        return res, memory, agreements

    def feedback(self, unique_id, turn, memory_ids, template_filling, agreements):
        res = self.chain_feedback.invoke(template_filling)["text"]
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
            "additionalArgs": template_filling
        }
        self.coordinator.updateGlobalMemory(unique_id, turn, self.id, self.persona, "feedback", res, agreements[-1]["agreement"],
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
                str(unique_id)] = f'''{{"messageId": {unique_id}, "turn": {turn}, "agentId": "{agent_id}", "persona": "{str(persona).replace('"', "'")}", "additionalArgs": {prompt_args}, "contribution": "{contribution}", "memoryIds": {memory_ids}, "text": "{str(text).replace('"', "'")}", "agreement": {agreement}, "extractedDraft": "{str(extracted_draft).replace('"', "'")}"}}'''
        self.saveMemoryToJson()

    def getMemory(self, context_length=None, turn=None, include_this_turn=True, extract_draft=False):
        '''
        Retrieves memory from the agents memory bucket as a dictionary
        Returns: dict
        '''
        memory = []
        memory_ids = []
        current_draft = None
        if os.path.exists(self.memory_bucket + ".dat"):
            with dbm.open(self.memory_bucket, 'r') as db:
                for key in db.keys():
                    memory.append(ast.literal_eval(db[key].decode().replace("\n", "\\n").replace("\t", "\\t"))) #TODO: Maybe reverse sort
            #memory = sorted(memory.items(), key=lambda x: x["messageId"], reverse=False)
            context_memory = []
            for m in memory:
                if context_length:
                    if m["turn"] >= turn - context_length:
                        if turn > m["turn"] or include_this_turn:
                            context_memory.append(m)
                            memory_ids.append(int(m["messageId"]))
                            if m["contribution"] == "draft" or (
                                    m["contribution"] == "improve" and "disagree" in m["text"].lower()):
                                current_draft = m["text"]
                else:
                    context_memory.append(m)
                    memory_ids.append(int(m["messageId"]))
                    if m["contribution"] == "draft" or (
                            m["contribution"] == "improve" and "disagree" in m["text"].lower()):
                        current_draft = m["text"]

            #context_memory = dict(context_memory)
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
            for m in memory:
                if m["persona"] != self.persona:
                    memory_string = memory_string + f"\n[INST]{m["persona"]}: {m["text"]}[/INST]"
                else:
                    memory_string = memory_string + f"\n{m["persona"]}: {m["text"]}"
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