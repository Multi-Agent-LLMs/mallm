import logging
from collections import Counter
from typing import Any, Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionAlteration, DecisionProtocol
from mallm.utils.prompts import (
    generate_approval_voting_prompt,
)
from mallm.utils.types import Agreement, VotingResult, VotingResults

logger = logging.getLogger("mallm")


class ApprovalVoting(DecisionProtocol):
    """
    The Approval Voting decision protocol allows panelists to approve any number of solutions after a certain number of turns.
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

        final_answers_with_confidence, voting_process_string = (
            self.generate_final_answers(agreements, question, task)
        )
        decision, final_answer, results, voting_process_string = (
            self.vote_with_alterations(
                final_answers_with_confidence,
                question,
                task,
                voting_process_string,
                "approval",
                generate_approval_voting_prompt,
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
        votes: list[int],
    ) -> dict[str, VotingResult]:
        # Count approvals for each answer
        approval_counts = Counter(votes)
        most_approved = approval_counts.most_common(1)[0][0]

        all_votes[alteration.value] = VotingResult(
            votes=votes,
            most_voted=most_approved,
            final_answer=final_answers[most_approved],
            agreed=True,
        )
        logger.info(
            f"Most approved answer from agent {self.panelists[most_approved].short_id}"
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
        approval_list = [
            int(a.strip())
            for a in vote_str.split(",")
            if 0 <= int(a.strip()) < len(final_answers)
        ]
        if approval_list:
            vote.extend(approval_list)
            logger.info(
                f"{panelist.persona} approved answers from {[self.panelists[a].persona for a in approval_list]}"
            )
            voting_process_string += f"{panelist.persona} approved answers from {[self.panelists[a].persona for a in approval_list]}\n"
            success = True
        return vote_str, vote, success, voting_process_string
