import logging
from collections import Counter

from mallm.agents.panelist import Panelist
from mallm.decision_making.DecisionProtocol import DecisionProtocol
from mallm.prompts.agent_prompts import (
    generate_final_answer_prompt,
    generate_voting_prompt,
)
from mallm.utils.types import Agreement

logger = logging.getLogger("mallm")


class Voting(DecisionProtocol):
    """
    The Voting decision protocol allows panelists to vote for the best answer after a certain number of turns.
    """

    def __init__(
        self, panelists: list[Panelist], use_moderator: bool, vote_turn: int = 3
    ):
        super().__init__(panelists, use_moderator)
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

        votes = []
        for panelist in self.panelists:
            while True:
                # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                vote = panelist.llm.invoke(
                    generate_voting_prompt(
                        panelist.persona,
                        panelist.persona_description,
                        task,
                        question,
                        final_answers,
                    )
                )
                try:
                    vote_int = int(vote.strip())
                    if 0 <= vote_int < len(final_answers):
                        votes.append(vote_int)
                        logger.info(
                            f"{panelist.short_id} voted for answer from {self.panelists[vote_int].short_id}"
                        )
                        break
                    else:
                        logger.debug(
                            f"{panelist.short_id} cast an invalid vote: {vote}. Asking to vote again."
                        )
                except ValueError:
                    logger.debug(
                        f"{panelist.short_id} cast an invalid vote: {vote}. Asking to vote again."
                    )

        # Search for the answer with the most votes from the agents
        vote_counts = Counter(votes)
        most_voted = vote_counts.most_common(1)[0][0]
        logger.info(
            f"Voted for answer from agent {self.panelists[most_voted].short_id}"
        )
        return final_answers[most_voted], True
