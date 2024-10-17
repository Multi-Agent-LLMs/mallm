import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mallm.agents.panelist import Panelist
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.config import Config
from mallm.utils.enums import DecisionAlteration
from mallm.utils.types import Agreement, VotingResult, VotingResultList, WorkerFunctions

logger = logging.getLogger("mallm")


class VotingPromptFunction(Protocol):
    def __call__(
        self,
        panelist: Panelist,
        panelists: list[Panelist],
        task: str,
        question: str,
        solutions: list[str],
        additional_context: Optional[str] = None,
        anonymous: bool = True,
        confidence: Optional[list[int]] = None,
        history: bool = False,
    ) -> list[dict[str, str]]: ...


class DecisionProtocol(ABC):
    """
    Abstract base class for a decision protocol in a multi-agent LLM framework.
    Any concrete decision protocol must implement the make_decision method.
    """

    _name = "DecisionProtocol"

    def __init__(
        self,
        panelists: list[Panelist],
        num_neutral_agents: int,
        worker_functions: WorkerFunctions,
    ) -> None:
        self.panelists: list[Panelist] = panelists
        self.num_neutral_agents: int = num_neutral_agents
        self.total_agents: int = len(panelists) + num_neutral_agents
        self.worker_functions = worker_functions

    def generate_final_answers(
        self, agreements: list[Agreement], question: str, task: str
    ) -> tuple[list[tuple[str, int]], str]:
        final_answers_with_confidence = []
        voting_process_string = ""
        for panelist in self.panelists:
            prev_answer: Agreement = next(
                a for a in agreements if a.agent_id == panelist.id
            )
            confidence = 0.0

            def confidence_callback(confidence_value: float) -> None:
                nonlocal confidence
                confidence = confidence_value

            response = panelist.llm.invoke(
                ResponseGenerator.generate_final_answer_prompt(
                    question,
                    task,
                    prev_answer.solution,
                ),
                confidence_callback=confidence_callback,
            )
            prev_answer.solution = response
            final_answers_with_confidence.append((response, int(confidence * 100)))
            voting_process_string += f"{panelist.persona} final answer: {response}\n"
        return final_answers_with_confidence, voting_process_string

    def vote_with_alterations(
        self,
        final_answers_with_confidence: list[tuple[str, int]],
        question: str,
        task: str,
        voting_process_string: str,
        decision_protocol_name: str,
        voting_prompt_function: VotingPromptFunction,
        alterations_enabled: bool = False,
        panelists: Optional[list[Panelist]] = None,
    ) -> tuple[bool, str, VotingResultList, str]:
        if panelists is None:
            panelists = self.panelists
        all_votes: dict[str, VotingResult] = {}
        facts = None
        final_answers = [answer for answer, _ in final_answers_with_confidence]
        confidences_static = []
        confidences_log_prob = [
            log_prob for _, log_prob in final_answers_with_confidence
        ]
        confidences_prompted = []
        confidences_consistency = []

        for alteration in (
            list(DecisionAlteration)
            if alterations_enabled
            else [DecisionAlteration.ANONYMOUS]
        ):
            voting_process_string += f"\nVoting with alteration: {alteration.value}\n"
            if alteration == DecisionAlteration.FACTS:
                facts = self.worker_functions.worker_context_function(question)
                voting_process_string += f"\nFacts: {facts}\n\n"
            if alteration == DecisionAlteration.CONFIDENCE:
                confidences_static = [100 for _ in self.panelists]
                voting_process_string += f"\nConfidence: {confidences_static}\n"
            if alteration == DecisionAlteration.CONFIDENCE_LOG_PROBS:
                voting_process_string += f"\nConfidence: {confidences_log_prob}\n"
            if alteration == DecisionAlteration.CONFIDENCE_PROMPTED:
                confidences_prompted = self.generate_prompted_confidence(
                    final_answers, question, task
                )
                voting_process_string += f"\nConfidence: {confidences_prompted}\n"
            if alteration == DecisionAlteration.CONFIDENCE_CONSISTENCY:
                confidences_consistency = self.get_consistency_confidences()
                voting_process_string += f"\nConfidence: {confidences_consistency}\n"
            votes: Any = []
            for panelist in panelists:
                retries = 0
                while retries < 10:
                    # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                    if alteration == DecisionAlteration.ANONYMOUS:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                            )
                        )
                    elif alteration == DecisionAlteration.FACTS:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                additional_context=facts,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                confidence=confidences_static,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE_LOG_PROBS:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                confidence=confidences_log_prob,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE_PROMPTED:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                confidence=confidences_prompted,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE_CONSISTENCY:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                confidence=confidences_consistency,
                            )
                        )
                    elif alteration == DecisionAlteration.PUBLIC:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                anonymous=False,
                            )
                        )
                    elif alteration == DecisionAlteration.HISTORY:
                        vote = panelist.llm.invoke(
                            voting_prompt_function(
                                panelist=panelist,
                                panelists=self.panelists,
                                task=task,
                                question=question,
                                solutions=final_answers,
                                history=True,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unknown DecisionAlteration type: {alteration.value}"
                        )
                    try:
                        vote, votes, success, voting_process_string = (
                            self.process_votes(
                                final_answers,
                                panelist,
                                vote,
                                votes,
                                voting_process_string,
                            )
                        )
                        if success:
                            break
                        raise ValueError
                    except (ValueError, json.JSONDecodeError):
                        retries += 1
                        logger.debug(
                            f"{panelist.short_id} provided an invalid vote: {vote}. Asking to re-vote."
                        )
                if retries >= 10:
                    logger.warning(
                        f"{panelist.short_id} reached maximum retries. Counting as invalid vote."
                    )

            all_votes = self.process_results(
                all_votes, alteration, final_answers, votes
            )
        results = VotingResultList(
            voting_process_string=voting_process_string,
            answers=final_answers,
            alterations=all_votes,
            type=decision_protocol_name,
        )
        final_answer: str = all_votes[DecisionAlteration.ANONYMOUS.value].final_answer
        decision: bool = all_votes[DecisionAlteration.ANONYMOUS.value].agreed
        return decision, final_answer, results, voting_process_string

    def get_consistency_confidences(self) -> list[int]:
        confidences_consistency = []
        for panelist in self.panelists:
            answers = panelist.get_own_messages()
            embeddings = self.worker_functions.worker_paraphrase_function(answers)

            cosine_sim_matrix = cosine_similarity(embeddings)

            # We only need the upper triangle of the matrix, excluding the diagonal
            upper_triangle_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
            pairwise_similarities = cosine_sim_matrix[upper_triangle_indices]

            # Calculate the average similarity (confidence score)
            confidence_score = np.mean(pairwise_similarities)
            confidences_consistency.append(int(confidence_score * 100))
        return confidences_consistency

    def generate_prompted_confidence(
        self, final_answers: list[str], question: str, task: str
    ) -> list[int]:
        confidences_prompted = []
        for final_answer, panelist in zip(final_answers, self.panelists):
            retries = 0
            confidence_score = None
            while retries < 10:
                confidence_prompted = panelist.llm.invoke(
                    ResponseGenerator.generate_answer_confidence_prompt(
                        panelist, question, task, final_answer
                    )
                )
                try:
                    confidence_score = int(confidence_prompted.strip())
                    if 0 <= confidence_score <= 100:
                        break
                except ValueError:
                    pass

                retries += 1
            if confidence_score is None or not (0 <= confidence_score <= 100):
                confidence_score = 0
            confidences_prompted.append(confidence_score)
        return confidences_prompted

    @abstractmethod
    def make_decision(
        self,
        agreements: list[Agreement],
        turn: int,
        agent_index: int,
        task: str,
        question: str,
        config: Config,
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResultList]]:
        """
        Abstract method to make a decision based on agreements, the current turn number, and the list of panelists.

        Parameters:
        agreements (list[dict[str, any]]): A list of agreement objects from agents.
        turn (int): The current turn number.

        Returns:
        str, bool: str is the result of the conversation and bool describes whether they agreed or not.
        """

    @abstractmethod
    def process_votes(
        self,
        final_answers: list[str],
        panelist: Panelist,
        vote_str: str,
        vote: Any,
        voting_process_string: str,
    ) -> tuple[str, Any, bool, str]:
        pass

    @abstractmethod
    def process_results(
        self,
        all_votes: dict[str, VotingResult],
        alteration: DecisionAlteration,
        final_answers: list[str],
        votes: Any,
    ) -> dict[str, VotingResult]:
        pass
