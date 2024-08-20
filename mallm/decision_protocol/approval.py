import logging
from collections import Counter
from typing import Optional

from contextplus import context

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
            approvals = []
            for panelist in self.panelists:
                retries = 0
                while retries < 10:
                    # Creates a prompt with all the answers and asks the agent to vote for the best one, 0 indexed inorder
                    if alteration == DecisionAlteration.ANONYMOUS:
                        approval = panelist.llm.invoke(
                            generate_approval_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                            )
                        )
                    elif alteration == DecisionAlteration.FACTS:
                        approval = panelist.llm.invoke(
                            generate_approval_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                additional_context=facts,
                            )
                        )
                    elif alteration == DecisionAlteration.CONFIDENCE:
                        approval = panelist.llm.invoke(
                            generate_approval_voting_prompt(
                                panelist,
                                self.panelists,
                                task,
                                question,
                                final_answers,
                                confidence=confidence,
                            )
                        )
                    elif alteration == DecisionAlteration.PUBLIC:
                        approval = panelist.llm.invoke(
                            generate_approval_voting_prompt(
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
                            voting_process_string += f"{panelist.persona} approved answers from {[self.panelists[a].persona for a in approval_list]}\n"
                            break
                        raise ValueError
                    except ValueError:
                        retries += 1
                        logger.debug(
                            f"{panelist.short_id} cast an invalid approval: {approval}. Asking to approve again. Retry {retries}/10."
                        )
                if retries >= 10:
                    logger.warning(
                        f"{panelist.short_id} reached maximum retries. Counting as invalid vote."
                    )

            # Count approvals for each answer
            approval_counts = Counter(approvals)
            most_approved = approval_counts.most_common(1)[0][0]

            all_votes[alteration.value] = VotingResult(
                votes=approvals,
                most_voted=most_approved,
                final_answer=final_answers[most_approved],
                agreed=True,
            )
            logger.info(
                f"Most approved answer from agent {self.panelists[most_approved].short_id}"
            )

        results = VotingResults(
            voting_process_string=voting_process_string,
            final_answers=final_answers,
            alterations=all_votes,
            type="approval",
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
