import logging

from mallm.agents.panelist import Panelist
from mallm.decision_making.decision_protocol import DecisionProtocol
from mallm.prompts.agent_prompts import (
    generate_final_answer_prompt,
    generate_ranking_prompt,
)
from mallm.utils.types import Agreement

logger = logging.getLogger("mallm")


class RankedVoting(DecisionProtocol):
    """
    The Ranked Voting decision protocol allows panelists to rank their preferences among a set of answers after a certain number of turns.
    """

    def __init__(self, panelists: list[Panelist], vote_turn: int = 3):
        super().__init__(panelists)
        self.vote_turn = vote_turn

    def make_decision(
        self, agreements: list[Agreement], turn: int, task: str, question: str
    ) -> tuple[str, bool]:
        if turn < self.vote_turn:
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

        rankings = []
        for panelist in self.panelists:
            while True:
                # Creates a prompt with all the answers and asks the agent to rank them
                ranking = panelist.llm.invoke(
                    generate_ranking_prompt(
                        panelist.persona,
                        panelist.persona_description,
                        task,
                        question,
                        final_answers,
                    )
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
                        break
                    else:
                        logger.debug(
                            f"{panelist.short_id} cast an invalid ranking: {ranking}. Asking to rank again."
                        )
                except ValueError:
                    logger.debug(
                        f"{panelist.short_id} cast an invalid ranking: {ranking}. Asking to rank again."
                    )

        # Calculate the score for each answer based on the rankings
        scores = [0] * len(final_answers)
        for ranking in rankings:
            for i, rank in enumerate(ranking):
                scores[rank] += 5 - i  # Score 5 for the 1st rank, 4 for the 2nd, etc.

        # Find the answer with the highest score
        highest_score = max(scores)
        best_answers = [
            final_answers[i] for i, score in enumerate(scores) if score == highest_score
        ]

        # If there's a tie, pick the latest answer among the best
        result = best_answers[-1] if best_answers else ""
        agreed = len(best_answers) == 1

        return result, agreed
