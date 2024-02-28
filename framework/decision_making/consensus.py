import fire

class Consensus():
    def __init__(self, agents, consensus_threshold, use_moderator = False):
        self.consensus_threshold = consensus_threshold
        self.agents = agents
        self.use_moderator = use_moderator

    def decide(self, agreements):
        if self.consensus_threshold:
            return len([a for a in agreements if a is True]) >= self.consensus_threshold
        else:
            return len([a for a in agreements if a is True]) >= len(self.agents) - self.use_moderator   # all (except moderator) have to agree

class MajorityConsensus():
    '''
    The Majority Consensus imitates the implementation by Yin et. al.
    Paper: https://arxiv.org/abs/2312.01823
    '''
    def __init__(self, agents, use_moderator = False):
        self.agents = agents
        self.use_moderator = use_moderator

    def decide(self, agreements, turn):
        if len(self.agents) - self.use_moderator == 3 and turn < 5:
            return len([a for a in agreements if a is True]) == 3   # all agents need to agree in the first 5 turns (except draft proposer)
        else:
            return len([a for a in agreements if a is True]) > (len(self.agents) - self.use_moderator)/2  # more than half of the agents need to agree (except moderator and draft proposer)
        
def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)