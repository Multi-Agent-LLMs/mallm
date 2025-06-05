import logging
from typing import Any, Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocols.protocol import DecisionProtocol
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.config import Config
from mallm.utils.enums import DecisionAlteration
from mallm.utils.types import Agreement, VotingResult, VotingResultList, WorkerFunctions

logger = logging.getLogger("mallm")


class Judge(DecisionProtocol):
    """
    The Judge decision protocol creates a summary of all answers after a certain number of turns.
    """

    _name = "summary"

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
                ResponseGenerator.generate_summary_prompt,
                config.voting_protocols_with_alterations,
                panelists=[self.panelists[0]],
            )
        )
        return (
            final_answer,
            decision,
            agreements,
            voting_process_string,
            results,
        )

    def process_results(  # noqa: PLR6301
        self,
        all_votes: dict[str, VotingResult],
        alteration: DecisionAlteration,
        final_answers: list[str],
        votes: Any,
    ) -> dict[str, VotingResult]:
        all_votes[alteration.value] = VotingResult(
            votes=None,
            most_voted=-1,
            final_answer=votes,
            agreed=True,
        )
        return all_votes

    def process_votes(  # noqa: PLR6301
        self,
        final_answers: list[str],
        panelist: Panelist,
        vote_str: str,
        vote: Any,
        voting_process_string: str,
    ) -> tuple[str, Any, bool, str]:
        logger.info(f"Created summary of all answers: {vote_str}")
        voting_process_string += f"Created summary of all answers: {vote_str}\n"
        return vote_str, vote_str, True, voting_process_string
