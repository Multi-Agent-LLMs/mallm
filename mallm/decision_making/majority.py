from typing import Optional

from mallm.agents.panelist import Panelist
from mallm.decision_making.decision_protocol import DecisionProtocol
from mallm.utils.types import Agreement
import logging

logger = logging.getLogger("mallm")


class ThresholdConsensus(DecisionProtocol):
    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
        threshold_percent: float = 0.5,
        threshold_turn: Optional[int] = None,
        threshold_agents: Optional[int] = None,
    ) -> None:
        super().__init__(panelists, use_moderator)
        self.threshold_turn = threshold_turn
        self.threshold_agents = threshold_agents
        self.threshold_percent = threshold_percent

    def make_decision(
        self,
        agreements: list[Agreement],
        turn: int,
        agent_index: int,
        task: str,
        question: str,
    ) -> tuple[str, bool]:
        reversed_agreements = agreements[::-1]
        # latest disagreement is the current draft
        num_agreements, current_agreement = next(
            (
                (i, agreement)
                for i, agreement in enumerate(reversed_agreements)
                if not agreement.agreement
            ),
            (None, None),
        )
        if not current_agreement or not num_agreements:
            return "", False

        if (self.threshold_agents and len(self.panelists) <= self.threshold_agents) or (
            self.threshold_turn and turn < self.threshold_turn
        ):
            # all agents need to agree in the first <threshold_turn> turns
            # all agents need to agree if there are less than <threshold_agents> agents
            return (
                current_agreement.response,
                num_agreements + 1 == self.total_agents,
            )
        else:
            # more than <threshold_percent> of the agents need to agree
            return (
                current_agreement.response,
                num_agreements + 1 > self.total_agents * self.threshold_percent,
            )


class MajorityConsensus(ThresholdConsensus):
    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
    ):
        super().__init__(panelists, use_moderator, 0.5, None, None)


class UnanimousConsensus(ThresholdConsensus):
    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
    ):
        super().__init__(panelists, use_moderator, 1.0, None, None)


class SupermajorityConsensus(ThresholdConsensus):
    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
    ):
        super().__init__(panelists, use_moderator, 0.66, None, None)


class HybridMajorityConsensus(ThresholdConsensus):
    """
    The Hybrid Majority Consensus imitates the implementation by Yin et. al.
    Paper: https://arxiv.org/abs/2312.01823
    """

    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
    ):
        super().__init__(panelists, use_moderator, 0.75, 5, 3)
