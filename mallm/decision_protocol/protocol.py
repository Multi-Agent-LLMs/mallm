from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from mallm.agents.panelist import Panelist
from mallm.utils.prompts import generate_final_answer_prompt
from mallm.utils.types import Agreement, VotingResults


class DecisionAlteration(Enum):
    PUBLIC = "public"
    FACTS = "facts"
    CONFIDENCE = "confidence"
    ANONYMOUS = "anonymous"


class DecisionProtocol(ABC):
    """
    Abstract base class for a decision protocol in a multi-agent LLM framework.
    Any concrete decision protocol must implement the make_decision method.
    """

    def __init__(self, panelists: list[Panelist], use_moderator: bool) -> None:
        self.panelists: list[Panelist] = panelists
        self.use_moderator: bool = use_moderator
        self.total_agents: int = len(panelists) + (1 if use_moderator else 0)

    def generate_final_answers(
        self, agreements: list[Agreement], question: str, task: str
    ):
        final_answers = []
        voting_process_string = ""
        for panelist in self.panelists:
            prev_answer: Agreement = next(
                a for a in agreements if a.agent_id == panelist.id
            )
            response = panelist.llm.invoke(
                generate_final_answer_prompt(
                    panelist.persona,
                    panelist.persona_description,
                    question,
                    task,
                    prev_answer.solution,
                )
            )
            prev_answer.solution = response
            final_answers.append(response)
            voting_process_string += f"{panelist.persona} final answer: {response}\n"
        return final_answers, voting_process_string

    @abstractmethod
    def make_decision(
        self,
        agreements: list[Agreement],
        turn: int,
        agent_index: int,
        task: str,
        question: str,
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResults]]:
        """
        Abstract method to make a decision based on agreements, the current turn number, and the list of panelists.

        Parameters:
        agreements (list[dict[str, any]]): A list of agreement objects from agents.
        turn (int): The current turn number.

        Returns:
        str, bool: str is the result of the conversation and bool describes whether they agreed or not.
        """
