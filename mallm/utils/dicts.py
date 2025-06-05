from mallm.decision_protocols.approval_voting import ApprovalVoting
from mallm.decision_protocols.consensus import (
    HybridMajorityConsensus,
    MajorityConsensus,
    SupermajorityConsensus,
    UnanimityConsensus,
)
from mallm.decision_protocols.consensus_voting import ConsensusVoting
from mallm.decision_protocols.cumulative_voting import CumulativeVoting
from mallm.decision_protocols.protocol import DecisionProtocol
from mallm.decision_protocols.ranked_voting import RankedVoting
from mallm.decision_protocols.simple_voting import SimpleVoting
from mallm.decision_protocols.summary import Summary
from mallm.discussion_paradigms.collective_refinement import CollectiveRefinement
from mallm.discussion_paradigms.debate import DiscussionDebate
from mallm.discussion_paradigms.memory import DiscussionMemory
from mallm.discussion_paradigms.paradigm import DiscussionParadigm
from mallm.discussion_paradigms.relay import DiscussionRelay
from mallm.discussion_paradigms.report import DiscussionReport
from mallm.models.discussion.CriticalResponseGenerator import CriticalResponseGenerator
from mallm.models.discussion.FreeTextResponseGenerator import FreeTextResponseGenerator
from mallm.models.discussion.ReasoningResponseGenerator import (
    ReasoningResponseGenerator,
)
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

DISCUSSION_PARADIGMS: dict[str, type[DiscussionParadigm]] = {
    "memory": DiscussionMemory,
    "report": DiscussionReport,
    "relay": DiscussionRelay,
    "debate": DiscussionDebate,
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
    "critical": CriticalResponseGenerator,
    "reasoning": ReasoningResponseGenerator,
}
