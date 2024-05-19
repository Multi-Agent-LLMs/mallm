from mallm.decision_making.DecisionProtocol import DecisionProtocol


class MajorityConsensus(DecisionProtocol):
    """
    The Majority Consensus imitates the implementation by Yin et. al.
    Paper: https://arxiv.org/abs/2312.01823
    """

    def make_decision(self, agreements, turn, task, question):
        if len(self.panelists) <= 3 and turn < 5:
            # all agents need to agree in the first 5 turns (except moderator)
            return [agreement for agreement in agreements if agreement.agreement][
                -1
            ].response, sum([a.agreement for a in agreements]) == len(self.panelists)
        else:
            # more than half of the agents need to agree (except moderator)
            return (
                [agreement for agreement in agreements if agreement.agreement][
                    -1
                ].response,
                sum([a.agreement for a in agreements]) > len(self.panelists) / 2,
            )
