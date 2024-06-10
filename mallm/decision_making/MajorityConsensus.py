from mallm.agents.panelist import Panelist
from mallm.decision_making.DecisionProtocol import DecisionProtocol
from mallm.utils.types import Agreement
import logging

logger = logging.getLogger("mallm")


class MajorityConsensus(DecisionProtocol):
    """
    The Majority Consensus imitates the implementation by Yin et. al.
    Paper: https://arxiv.org/abs/2312.01823
    """

    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
        majority_turn: int = 5,
        majority_agents: int = 3,
    ):
        super().__init__(panelists, use_moderator)
        self.majority_turn = majority_turn
        self.majority_agents = majority_agents

    def make_decision(
        self, agreements: list[Agreement], turn: int, task: str, question: str
    ) -> tuple[str, bool]:
        min_agents = len(self.panelists)
        if self.use_moderator:
            # remove all agreements before the last moderator draft
            agreements = agreements[
                next(
                    (
                        i
                        for i, agreement in reversed(list(enumerate(agreements)))
                        if agreement.agreement == None
                    ),
                    -1,
                )
                + 1 :
            ]
        if not self.use_moderator:
            # remove all agreements before the last disagreement
            agreements = agreements[
                next(
                    (
                        i
                        for i, agreement in reversed(list(enumerate(agreements)))
                        if not agreement.agreement
                    ),
                    -1,
                )
                + 1 :
            ]
            min_agents -= 1
        if len(self.panelists) <= self.majority_agents and turn < self.majority_turn:
            # all agents need to agree in the first <majority_turn> turns
            agents_agree = [
                agreement for agreement in agreements if agreement.agreement
            ]

            if not agents_agree:
                return "", False

            return (
                agents_agree[-1].response,
                len([a.agreement for a in agreements if a.agreement]) == min_agents,
            )
        else:
            # more than half of the agents need to agree
            agents_agree = [
                agreement for agreement in agreements if agreement.agreement
            ]

            if not agents_agree:
                return "", False

            return (
                agents_agree[
                    -1
                ].response,  # TODO: It is misleading how this is a generic [AGREE] string rather than the most recent draft
                len([a.agreement for a in agreements if a.agreement]) > min_agents / 2,
            )
