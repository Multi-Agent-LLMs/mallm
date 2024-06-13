from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    data: str
    out: str
    instruction: str
    endpoint_url: str = "https://api.openai.com"
    model: str = "gpt-3.5-turbo"
    api_key: str = "-"
    use_moderator: bool = False
    max_turns: int = 10
    force_all_turns: bool = False
    feedback_sentences: Optional[tuple[int, int]] = None
    paradigm: str = "memory"
    decision_protocol: str = "hybrid_consensus"
    context_length: int = 3
    include_current_turn_in_memory: bool = True
    extract_all_drafts: bool = True
    debate_rounds: Optional[int] = None
    max_concurrent_requests: int = 100
    clear_memory_bucket: bool = True
    memory_bucket_dir: str = "./mallm/utils/memory_bucket/"
    baseline: bool = False
    chain_of_thought: bool = True
    num_agents: int = 3
    agent_generator: str = "expert"
