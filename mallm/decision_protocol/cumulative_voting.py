import ast
import logging
from typing import Any, Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionAlteration, DecisionProtocol
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.config import Config
from mallm.utils.types import Agreement, VotingResult, VotingResultList, WorkerFunctions

logger = logging.getLogger("mallm")


class CumulativeVoting(DecisionProtocol):
    """
    The Cumulative Voting decision protocol allows panelists to distribute 10 points among the solutions.
    The solution with the highest total points is selected as the final decision.
    """

    _name = "cumulative_voting"

    def __init__(
        self,
        panelists: list[Panelist],
        num_neutral_agents: int,
        worker_functions: WorkerFunctions,
        vote_turn: int = 3,
    ) -> None:
        super().__init__(panelists, num_neutral_agents, worker_functions)
        self.vote_turn = vote_turn

    def make_decision(
        self,
        agreements: list[Agreement],
        turn: int,
        agent_index: int,
        task: str,
        question: str,
        config: Config,
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResultList]]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]

        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False, agreements, "", None

        final_answers_with_confidence, voting_process_string = (
            self.generate_final_answers(agreements, question, task)
        )

        decision, final_answer, results, voting_process_string = (
            self.vote_with_alterations(
                final_answers_with_confidence,
                question,
                task,
                voting_process_string,
                self._name,
                ResponseGenerator.generate_cumulative_voting_prompt,
                config.voting_protocols_with_alterations,
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
        votes: Any,
    ) -> dict[str, VotingResult]:
        # Aggregate points for each solution
        total_points = [0] * len(final_answers)
        for points in votes:
            for index, point in points.items():
                total_points[index] += point
        # Determine the solution with the highest points, break ties by selecting the first solution and go for another round
        max_points = max(total_points)
        best_solution_index = total_points.index(max_points)
        best_answers = [
            final_answers[i]
            for i, score in enumerate(total_points)
            if score == max_points
        ]
        if len(best_answers) == 1:
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=best_solution_index,
                final_answer=final_answers[best_solution_index],
                agreed=True,
            )
            logger.info(
                f"Selected answer from agent {self.panelists[best_solution_index].short_id} with {max_points} points"
            )
        else:
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=-1,
                final_answer="",
                agreed=False,
            )
            logger.info("There was a tie. Going for another round of voting.")
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
        vote_str = vote_str.replace("\n", "").replace(" ", "").strip()
        points_dict = ast.literal_eval(vote_str)
        points_dict = {int(k): int(v) for k, v in points_dict.items()}
        if self.validate_points_distribution(points_dict, len(final_answers)):
            vote.append(points_dict)
            logger.info(f"{panelist.persona} allocated points: {points_dict}")
            voting_process_string += (
                f"{panelist.persona} allocated points: {points_dict}\n"
            )
            success = True
        return vote_str, vote, success, voting_process_string

    @staticmethod
    def validate_points_distribution(
        points_dict: dict[int, int], num_solutions: int
    ) -> bool:
        total_points = sum(points_dict.values())
        if total_points != 10:
            return False
        for index in points_dict:
            if not isinstance(index, int) or not (0 <= index < num_solutions):
                return False
        return not any(x < 0 for x in points_dict.values())
