import ast
import json
import logging

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.utils.prompts import (
    generate_approval_voting_prompt,
    generate_final_answer_prompt,
    generate_cumulative_voting_prompt,
)
from mallm.utils.types import Agreement

logger = logging.getLogger("mallm")


class CumulativeVoting(DecisionProtocol):
    """
    The Cumulative Voting decision protocol allows panelists to distribute 10 points among the solutions.
    The solution with the highest total points is selected as the final decision.
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
    ) -> tuple[str, bool]:
        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False
        final_answers = []
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
                    prev_answer.response,
                )
            )
            prev_answer.response = response
            final_answers.append(response)

        # Collect points distribution from each panelist
        point_distributions = []
        for panelist in self.panelists:
            while True:
                point_distribution = panelist.llm.invoke(
                    generate_cumulative_voting_prompt(
                        panelist.persona,
                        panelist.persona_description,
                        task,
                        question,
                        final_answers,
                    )
                )
                point_distribution = (
                    point_distribution.replace("\n", "").replace(" ", "").strip()
                )
                try:
                    points_dict = ast.literal_eval(point_distribution)
                    points_dict = {int(k): int(v) for k, v in points_dict.items()}
                    if self.validate_points_distribution(
                        points_dict, len(final_answers)
                    ):
                        point_distributions.append(points_dict)
                        logger.info(
                            f"{panelist.short_id} allocated points: {points_dict}"
                        )
                        break
                    else:
                        raise ValueError
                except (ValueError, json.JSONDecodeError):
                    logger.debug(
                        f"{panelist.short_id} provided an invalid points distribution: {point_distribution}. Asking to distribute points again."
                    )

        # Aggregate points for each solution
        total_points = [0] * len(final_answers)
        for points in point_distributions:
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
        agreed = len(best_answers) == 1

        logger.info(
            f"Selected answer from agent {self.panelists[best_solution_index].short_id} with {max_points} points"
        )

        return final_answers[best_solution_index], agreed

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
        if any(x < 0 for x in points_dict.values()):
            return False
        return True
