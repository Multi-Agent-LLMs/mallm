import logging
from collections import Counter

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.utils.prompts import (
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
    ) -> tuple[str, bool, list[Agreement], str]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]

        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False, agreements, ""
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
                    prev_answer.solution,
                )
            )
            prev_answer.solution = response
            final_answers.append(response)

        votes = []
        voting_process_string = ""
        for panelist in self.panelists:
            retries = 0
            while retries < 10:
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
                        voting_process_string += f"{panelist.persona} voted for answer from {self.panelists[vote_int].persona}\n"
                        break
                    logger.debug(
                        f"{panelist.short_id} cast an invalid vote: {vote}. Asking to vote again."
                    )
                except ValueError:
                    retries += 1
                    logger.debug(
                        f"{panelist.short_id} cast an invalid vote: {vote}. Asking to vote again."
                    )
            if retries >= 10:
                logger.warning(
                    f"{panelist.short_id} reached maximum retries. Counting as invalid vote."
                )

        # Search for the answer with the most votes from the agents
        vote_counts = Counter(votes)
        most_voted = vote_counts.most_common(1)[0][0]
        logger.info(
            f"Voted for answer from agent {self.panelists[most_voted].short_id}"
        )
        return final_answers[most_voted], True, agreements, voting_process_string
