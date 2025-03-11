import json
import logging
import random
from collections import Counter
from typing import Any, Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.config import Config
from mallm.utils.enums import DecisionAlteration
from mallm.utils.types import Agreement, VotingResult, VotingResultList, WorkerFunctions

logger = logging.getLogger("mallm")


class ConsensusVoting(DecisionProtocol):
    """
    The Voting decision protocol allows panelists to vote for the best answer after a certain number of turns.
    """

    _name = "consensus_voting"

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

        if agent_index != self.total_agents - 1:
            return "", False, agreements, "", None

        final_answers = []
        votes: Any = []
        all_votes: dict[str, VotingResult] = {}

        voting_process_string = ""
        for agreement in agreements:
            voting_process_string += f"{agreement.persona} final answer: {agreement.solution}\n"
            final_answers.append(agreement.solution)

        final_answers = self.remove_duplicate_answers(final_answers)

        for panelist in self.panelists:
            retries = 0
            while retries < 10:
                # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                vote = panelist.llm.invoke(
                    ResponseGenerator.generate_voting_prompt(
                        panelist=panelist,
                        panelists=self.panelists,
                        task=task,
                        question=question,
                        solutions=final_answers,
                    )
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
                all_votes, DecisionAlteration.PUBLIC, final_answers, votes
            )
        results = VotingResultList(
            voting_process_string=voting_process_string,
            answers=final_answers,
            alterations=all_votes,
            type=self._name,
        )
        final_answer: str = all_votes[DecisionAlteration.PUBLIC.value].final_answer
        decision: bool = all_votes[DecisionAlteration.PUBLIC.value].agreed

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
        votes: list[int],
    ) -> dict[str, VotingResult]:
        winners = []
        most_voted = -1
        if votes:
            vote_counts = Counter(votes)
            most_common = vote_counts.most_common()
            most_voted = most_common[0][0]
            most_voted_count = most_common[0][1]

            # Check if there are multiple winners with the same vote count
            winners = [
                candidate
                for candidate, count in vote_counts.items()
                if count == most_voted_count
            ]

        if len(winners) == 1:
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=most_voted,
                final_answer=final_answers[most_voted],
                agreed=True,
            )
            logger.info(
                f"Voted for answer from agent {self.panelists[most_voted].persona}"
            )
        else:
            random_voted = random.randint(0, len(final_answers) - 1)
            all_votes[alteration.value] = VotingResult(
                votes=votes,
                most_voted=random_voted,
                final_answer=final_answers[random_voted],
                agreed=False,
            )
            logger.info("There was a tie. Selecting a random solution. agreed=false marks this unsuccessful decision.")
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
        vote_int = int("".join([x for x in vote_str if x.isnumeric()]))

        # if len(final_answers) == 1:    # TODO: Add this in a future PR
        #    vote_int = 0    # If there is only one answer, the agent must vote for it

        if 0 <= vote_int < len(final_answers):
            vote.append(vote_int)
            logger.info(
                f"{panelist.persona} voted for answer from {self.panelists[vote_int].persona}"
            )
            voting_process_string += f"{panelist.persona} voted for answer from {self.panelists[vote_int].persona}\n"
            success = True
        return vote_str, vote, success, voting_process_string
