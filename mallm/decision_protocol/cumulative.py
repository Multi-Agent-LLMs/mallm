import ast
import json
import logging
from typing import Optional

from contextplus import context

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionAlteration, DecisionProtocol
from mallm.utils.prompts import (
    generate_cumulative_voting_prompt,
)
from mallm.utils.types import Agreement, VotingResult, VotingResults

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
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResults]]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]

        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False, agreements, "", None

        final_answers, voting_process_string = self.generate_final_answers(
            agreements, question, task
        )

        # Collect points distribution from each panelist
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
            point_distributions = []
            for panelist in self.panelists:
                retries = 0
                while retries < 10:
                    # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                    if alteration == DecisionAlteration.ANONYMOUS:
                        point_distribution = panelist.llm.invoke(
                            generate_cumulative_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                            )
                        )
                    elif alteration == DecisionAlteration.FACTS:
                        point_distribution = panelist.llm.invoke(
                            generate_cumulative_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                additional_context=facts,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE:
                        point_distribution = panelist.llm.invoke(
                            generate_cumulative_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                confidence=confidence,
                            )
                        )
                    elif alteration == DecisionAlteration.PUBLIC:
                        point_distribution = panelist.llm.invoke(
                            generate_cumulative_voting_prompt(
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
                            voting_process_string += (
                                f"{panelist.persona} allocated points: {points_dict}\n"
                            )
                            break
                        raise ValueError
                    except (ValueError, json.JSONDecodeError):
                        retries += 1
                        logger.debug(
                            f"{panelist.short_id} provided an invalid points distribution: {point_distribution}. Asking to distribute points again."
                        )
                if retries >= 10:
                    logger.warning(
                        f"{panelist.short_id} reached maximum retries. Counting as invalid vote."
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

            all_votes[alteration.value] = VotingResult(
                votes=point_distributions,
                most_voted=best_solution_index,
                final_answer=final_answers[best_solution_index],
                agreed=agreed,
            )
            logger.info(
                f"Selected answer from agent {self.panelists[best_solution_index].short_id} with {max_points} points"
            )

        results = VotingResults(
            voting_process_string=voting_process_string,
            final_answers=final_answers,
            alterations=all_votes,
            type="cumulative",
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
