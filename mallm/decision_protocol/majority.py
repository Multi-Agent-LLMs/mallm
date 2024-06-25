import logging
from typing import Optional

from mallm.agents.panelist import Panelist
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.utils.types import Agreement

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
    ) -> tuple[str, bool, list[Agreement]]:
        if len(agreements) > self.total_agents:
            agreements = agreements[-self.total_agents :]
        reversed_agreements = agreements[::-1]

        num_agreements, current_agreement = next(
            (
                (i, agreement)
                for i, agreement in enumerate(reversed_agreements)
                if not agreement.agreement
            ),
            (None, None),
        )

        # so we have at least some output if the discussion does not converge
        if not current_agreement or not num_agreements:
            moderator_agreement = next(
                (a for a in reversed_agreements if a.agreement is None), None
            )
            if moderator_agreement:
                return moderator_agreement.solution, False, agreements
            try:
                recent_panelist_agreement = next(
                    a for a in reversed_agreements if not a.agreement
                )
            except StopIteration:
                logger.warning(
                    "Failed gathering the most recent disagreement. Returning None as current solution."
                )
                return "None", False, agreements
            return recent_panelist_agreement.solution, False, agreements

        if (self.threshold_agents and len(self.panelists) <= self.threshold_agents) or (
            self.threshold_turn and turn < self.threshold_turn
        ):
            # all agents need to agree in the first <threshold_turn> turns
            # all agents need to agree if there are less than <threshold_agents> agents
            return (
                current_agreement.solution,
                num_agreements + 1 == self.total_agents,
                agreements,
            )
        # more than <threshold_percent> of the agents need to agree
        return (
            current_agreement.solution,
            num_agreements + 1 > self.total_agents * self.threshold_percent,
            agreements,
        )


class MajorityConsensus(ThresholdConsensus):
    def __init__(
        self,
        panelists: list[Panelist],
        use_moderator: bool,
    ):
        super().__init__(panelists, use_moderator, 0.5, None, None)


class UnanimityConsensus(ThresholdConsensus):
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
