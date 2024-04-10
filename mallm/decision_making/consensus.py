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
            return sum(agreements) == len(
                self.panelists)  # all agents need to agree in the first 5 turns (except moderator)
        else:
            return sum(agreements) > len(
                self.panelists) / 2  # more than half of the agents need to agree (except moderator)


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)
