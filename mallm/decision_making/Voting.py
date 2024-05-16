from mallm.decision_making.DecisionProtocol import DecisionProtocol
from mallm.prompts.agent_prompts import generate_final_answer_prompt


class Voting(DecisionProtocol):
    """
    The Majority Consensus imitates the implementation by Yin et. al.
    Paper: https://arxiv.org/abs/2312.01823
    """

    def make_decision(self, agreements, turn):
        for panelist in self.panelists:
            panelist.llm.invoke(
                generate_final_answer_prompt(
                    panelist.persona, panelist.persona_description
                )
            )
        final_answers = [a.response for a in agreements]

        if len(self.panelists) <= 3 and turn < 5:
            # all agents need to agree in the first 5 turns (except moderator)
            return agreements[-1].response, sum(
                [a.agreement for a in agreements]
            ) == len(self.panelists)
        else:
            # more than half of the agents need to agree (except moderator)
            return (
                agreements[-1].response,
                sum([a.agreement for a in agreements]) > len(self.panelists) / 2,
            )
