import logging
from typing import Any, Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionAlteration, DecisionProtocol
from mallm.utils.prompts import (
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

        final_answers, voting_process_string = self.generate_final_answers(
            agreements, question, task
        )

        decision, final_answer, results, voting_process_string = (
            self.vote_with_alterations(
                final_answers,
                question,
                task,
                voting_process_string,
                "ranking",
                generate_ranking_prompt,
            )
        )
        return (
            final_answer,
            decision,
            agreements,
            voting_process_string,
            results,
        )

    def process_results(
        self,
        all_votes: dict[str, VotingResult],
        alteration: DecisionAlteration,
        final_answers: list[str],
        votes: list[list[int]],
    ) -> dict[str, VotingResult]:
        # Calculate the score for each answer based on the rankings
        scores = [0] * len(final_answers)
        for ranking_list in votes:
            for rank, idx in enumerate(ranking_list):
                scores[idx] += (
                    min(5, self.total_agents) - rank
                )  # Score 5 for the 1st rank, 4 for the 2nd, etc.

        # Find the answer with the highest score
        highest_score = max(scores)
        index = scores.index(highest_score)
        best_answers = [
            final_answers[i] for i, score in enumerate(scores) if score == highest_score
        ]

        # If there's a tie, pick the first answer among the best
        # If all panelists agree on the best answer finished else go for another round
        agreed = len(best_answers) == 1
        all_votes[alteration.value] = VotingResult(
            votes=votes,
            most_voted=index,
            final_answer=final_answers[index],
            agreed=agreed,
        )
        logger.info(
            f"Selected answer from agent {self.panelists[index].short_id} with {highest_score} points"
        )
        return all_votes

    def process_votes(
        self,
        final_answers: list[str],
        panelist: Panelist,
        vote_str: str,
        vote: Any,
        voting_process_string: str,
    ) -> tuple[str, Any, bool, str]:
        success = False
        # Split the ranking and convert to a list of integers
        ranking_list = list(map(int, vote_str.strip().split()))
        if (
            all(0 <= rank < len(final_answers) for rank in ranking_list)
            and len(ranking_list) <= 5
        ):
            vote.append(ranking_list)
            logger.info(
                f"{panelist.persona} ranked answers: {[self.panelists[a].persona for a in ranking_list]}"
            )
            voting_process_string += f"{panelist.persona} ranked answers: {[self.panelists[a].persona for a in ranking_list]}\n"
            success = True
        return vote_str, vote, success, voting_process_string
