from mallm.agents.panelist import Panelist
from mallm.coordinator import Coordinator
from mallm.decision_making.MajorityConsensus import MajorityConsensus

coordinator = Coordinator(None, None)
panelists = [Panelist(None, None, coordinator, f"Person{i}", "") for i in range(5)]


def test_unanimous_decision_in_first_five_turns():
    mc = MajorityConsensus(panelists[:3])
    agreements = [{"res": "agree", "agreement": 1} for _ in range(3)]
    decision, is_consensus = mc.make_decision(agreements, 4)
    assert is_consensus


def test_no_unanimous_decision_in_first_five_turns():
    mc = MajorityConsensus(panelists[:3])
    agreements = [{"res": "agree", "agreement": 1} for _ in range(2)]
    agreements.append({"res": "disagree", "agreement": 0})
    decision, is_consensus = mc.make_decision(agreements, 4)
    assert not is_consensus


def test_majority_decision_after_five_turns():
    mc = MajorityConsensus(panelists)
    agreements = [{"res": "agree", "agreement": 1} for _ in range(3)]
    agreements.extend([{"res": "disagree", "agreement": 0} for _ in range(2)])
    decision, is_consensus = mc.make_decision(agreements, 6)
    assert is_consensus


def test_no_majority_decision_after_five_turns():
    mc = MajorityConsensus(panelists)
    agreements = [{"res": "agree", "agreement": 1} for _ in range(2)]
    agreements.extend([{"res": "disagree", "agreement": 0} for _ in range(3)])
    decision, is_consensus = mc.make_decision(agreements, 6)
    assert not is_consensus
