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

def main():
    pass

if __name__ == "__main__":
    fire.Fire(main)