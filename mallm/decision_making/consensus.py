import fire


class MajorityConsensus:
    '''
    The Majority Consensus imitates the implementation by Yin et. al.
    Paper: https://arxiv.org/abs/2312.01823
    '''

    def __init__(self, panelists):
        self.panelists = panelists

    def decide(self, agreements, turn):
        if len(self.panelists) <= 3 and turn < 5:
            # all agents need to agree in the first 5 turns (except moderator)
            return sum([a["agreement"] for a in agreements]) == len(self.panelists)
        else:
            # more than half of the agents need to agree (except moderator)
            return sum([a["agreement"] for a in agreements]) > len(self.panelists) / 2


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
