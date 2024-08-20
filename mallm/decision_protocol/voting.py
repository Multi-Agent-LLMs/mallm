import logging
from collections import Counter
from typing import Optional

from contextplus import context

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionAlteration, DecisionProtocol
from mallm.utils.prompts import (
    generate_voting_prompt,
)
from mallm.utils.types import Agreement, VotingResult, VotingResults

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
    ) -> tuple[str, bool, list[Agreement], str, Optional[VotingResults]]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]

        if turn < self.vote_turn or agent_index != self.total_agents - 1:
            return "", False, agreements, "", None

        final_answers, voting_process_string = self.generate_final_answers(
            agreements, question, task
        )

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
            votes = []
            for panelist in self.panelists:
                retries = 0
                while retries < 10:
                    # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                    if alteration == DecisionAlteration.ANONYMOUS:
                        vote = panelist.llm.invoke(
                            generate_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                            )
                        )
                    elif alteration == DecisionAlteration.FACTS:
                        vote = panelist.llm.invoke(
                            generate_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                additional_context=facts,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE:
                        vote = panelist.llm.invoke(
                            generate_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                confidence=confidence,
                            )
                        )
                    elif alteration == DecisionAlteration.PUBLIC:
                        vote = panelist.llm.invoke(
                            generate_voting_prompt(
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
                        vote_int = int(vote.strip())
                        if 0 <= vote_int < len(final_answers):
                            votes.append(vote_int)
                            logger.info(
                                f"{panelist.short_id} voted for answer from {self.panelists[vote_int].short_id}"
                            )
                            voting_process_string += f"{panelist.persona} voted for answer from {self.panelists[vote_int].persona}\n"
                            break
                        raise ValueError
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
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=most_voted,
                final_answer=final_answers[most_voted],
                agreed=True,
            )
            logger.info(
                f"Voted for answer from agent {self.panelists[most_voted].short_id}"
            )
        results = VotingResults(
            voting_process_string=voting_process_string,
            final_answers=final_answers,
            alterations=all_votes,
            type="voting",
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
