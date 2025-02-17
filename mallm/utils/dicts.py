from mallm.decision_protocol.approval_voting import ApprovalVoting
from mallm.decision_protocol.consensus import (
    HybridMajorityConsensus,
    MajorityConsensus,
    SupermajorityConsensus,
    UnanimityConsensus,
)
from mallm.decision_protocol.consensus_voting import ConsensusVoting
from mallm.decision_protocol.cumulative_voting import CumulativeVoting
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.decision_protocol.ranked_voting import RankedVoting
from mallm.decision_protocol.simple_voting import SimpleVoting
from mallm.decision_protocol.summary import Summary
from mallm.discourse_policy.collective_refinement import CollectiveRefinement
from mallm.discourse_policy.debate import DiscourseDebate
from mallm.discourse_policy.memory import DiscourseMemory
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.discourse_policy.relay import DiscourseRelay
from mallm.discourse_policy.report import DiscourseReport
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.models.discussion.SimpleResponseGenerator import SimpleResponseGenerator
from mallm.models.discussion.SplitFreeTextResponseGenerator import (
    SplitFreeTextResponseGenerator,
)
from mallm.models.personas.ExpertGenerator import ExpertGenerator
from mallm.models.personas.IPIPPersonaGenerator import IPIPPersonaGenerator
from mallm.models.personas.MockGenerator import MockGenerator
from mallm.models.personas.NoPersonaGenerator import NoPersonaGenerator
from mallm.models.personas.PersonaGenerator import PersonaGenerator

DECISION_PROTOCOLS: dict[str, type[DecisionProtocol]] = {
    "majority_consensus": MajorityConsensus,
    "supermajority_consensus": SupermajorityConsensus,
    "hybrid_consensus": HybridMajorityConsensus,
    "unanimity_consensus": UnanimityConsensus,
    "simple_voting": SimpleVoting,
    "approval_voting": ApprovalVoting,
    "cumulative_voting": CumulativeVoting,
    "ranked_voting": RankedVoting,
    "consensus_voting": ConsensusVoting,
    "summary": Summary,
}

DISCUSSION_PARADIGMS: dict[str, type[DiscoursePolicy]] = {
    "memory": DiscourseMemory,
    "report": DiscourseReport,
    "relay": DiscourseRelay,
    "debate": DiscourseDebate,
    "collective_refinement": CollectiveRefinement,
}

PERSONA_GENERATORS: dict[str, type[PersonaGenerator]] = {
    "expert": ExpertGenerator,
    "ipip": IPIPPersonaGenerator,
    "nopersona": NoPersonaGenerator,
    "mock": MockGenerator,
}

RESPONSE_GENERATORS: dict[str, type[ResponseGenerator]] = {
    "freetext": FreeTextResponseGenerator,
    "splitfreetext": SplitFreeTextResponseGenerator,
    "simple": SimpleResponseGenerator,
}
