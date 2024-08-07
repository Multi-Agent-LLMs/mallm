import logging
from typing import Optional

from contextplus import context

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionAlteration, DecisionProtocol
from mallm.utils.prompts import (
    generate_final_answer_prompt,
    generate_ranking_prompt,
)
from mallm.utils.types import Agreement, VotingResult, VotingResults

logger = logging.getLogger("mallm")


class RankedVoting(DecisionProtocol):
    """
    The Ranked Voting decision protocol allows panelists to rank their preferences among a set of answers after a certain number of turns.
    """

    def __init__(
        self, panelists: list[Panelist], use_moderator: bool, vote_turn: int = 3
    ) -> None:
        super().__init__(panelists, use_moderator)
        self.vote_turn = vote_turn

    def make_decision(
        self,
        agreements: list[Agreement],
        turn: int,
        agent_index: int,
        task: str,
        question: str,
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResults]]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]

        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False, agreements, "", None

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

        all_votes = {}
        facts = None
        confidence = []
        for alteration in DecisionAlteration:
            voting_process_string += f"\nVoting with alteration: {alteration.value}\n"
            if alteration == DecisionAlteration.FACTS:
                facts = context(question)
                voting_process_string += f"\nFacts: {facts}\n\n"
            if alteration == DecisionAlteration.CONFIDENCE:
                confidence = [100.0 for _ in self.panelists]
                voting_process_string += f"\nConfidence: {confidence}\n"
            rankings = []
            for panelist in self.panelists:
                retries = 0
                while retries < 10:
                    # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                    if alteration == DecisionAlteration.ANONYMOUS:
                        ranking = panelist.llm.invoke(
                            generate_ranking_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                            )
                        )
                    elif alteration == DecisionAlteration.FACTS:
                        ranking = panelist.llm.invoke(
                            generate_ranking_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                additional_context=facts,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE:
                        ranking = panelist.llm.invoke(
                            generate_ranking_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                confidence=confidence,
                            )
                        )
                    elif alteration == DecisionAlteration.PUBLIC:
                        ranking = panelist.llm.invoke(
                            generate_ranking_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                anonymous=False,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unknown DecisionAlteration type: {alteration.value}"
                        )
                    try:
                        # Split the ranking and convert to a list of integers
                        ranking_list = list(map(int, ranking.strip().split()))
                        if (
                            all(0 <= rank < len(final_answers) for rank in ranking_list)
                            and len(ranking_list) <= 5
                        ):
                            rankings.append(ranking_list)
                            logger.info(
                                f"{panelist.short_id} ranked answers: {ranking_list}"
                            )
                            voting_process_string += (
                                f"{panelist.persona} ranked answers: {ranking_list}\n"
                            )
                            break
                        raise ValueError
                    except ValueError:
                        retries += 1
                        logger.debug(
                            f"{panelist.short_id} cast an invalid ranking: {ranking}. Asking to rank again."
                        )
                if retries >= 10:
                    logger.warning(
                        f"{panelist.short_id} reached maximum retries. Counting as invalid vote."
                    )

            # Calculate the score for each answer based on the rankings
            scores = [0] * len(final_answers)
            for ranking_list in rankings:
                for rank, idx in enumerate(ranking_list):
                    scores[idx] += (
                        min(5, self.total_agents) - rank
                    )  # Score 5 for the 1st rank, 4 for the 2nd, etc.

            # Find the answer with the highest score
            highest_score = max(scores)
            index = scores.index(highest_score)
            best_answers = [
                final_answers[i]
                for i, score in enumerate(scores)
                if score == highest_score
            ]

            # If there's a tie, pick the first answer among the best
            # If all panelists agree on the best answer finished else go for another round
            agreed = len(best_answers) == 1
            all_votes[alteration.value] = VotingResult(
                votes=rankings,
                most_voted=index,
                final_answer=final_answers[index],
                agreed=agreed,
            )
            logger.info(
                f"Selected answer from agent {self.panelists[index].short_id} with {highest_score} points"
            )
        results = VotingResults(
            voting_process_string=voting_process_string,
            final_answers=final_answers,
            alterations=all_votes,
            type="ranked",
        )
        final_answer: str = final_answers[
            all_votes[DecisionAlteration.ANONYMOUS.value].most_voted
        ]
        decision: bool = all_votes[DecisionAlteration.ANONYMOUS.value].agreed
        return (
            final_answer,
            decision,
            agreements,
            voting_process_string,
            results,
        )
