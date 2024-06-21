import logging
from collections import Counter

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.utils.prompts import (
    generate_approval_voting_prompt,
    generate_final_answer_prompt,
)
from mallm.utils.types import Agreement

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
                    prev_answer.solution,
                )
            )
            prev_answer.solution = response
            final_answers.append(response)

        approvals = []
        for panelist in self.panelists:
            while True:
                # Creates a prompt with all the answers and asks the agent to vote for all acceptable ones, 0 indexed inorder
                approval = panelist.llm.invoke(
                    generate_approval_voting_prompt(
                        panelist.persona,
                        panelist.persona_description,
                        task,
                        question,
                        final_answers,
                    )
                )
                try:
                    approval_list = [
                        int(a.strip())
                        for a in approval.split(",")
                        if 0 <= int(a.strip()) < len(final_answers)
                    ]
                    if approval_list:
                        approvals.extend(approval_list)
                        logger.info(
                            f"{panelist.short_id} approved answers from {[self.panelists[a].short_id for a in approval_list]}"
                        )
                        break
                    else:
                        raise ValueError
                except ValueError:
                    logger.debug(
                        f"{panelist.short_id} cast an invalid approval: {approval}. Asking to approve again."
                    )

        # Count approvals for each answer
        approval_counts = Counter(approvals)
        most_approved = approval_counts.most_common(1)[0][0]
        logger.info(
            f"Most approved answer from agent {self.panelists[most_approved].short_id}"
        )
        return final_answers[most_approved], True
