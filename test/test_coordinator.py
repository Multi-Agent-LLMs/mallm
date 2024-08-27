from unittest.mock import Mock

import pytest

from mallm.coordinator import Coordinator
from mallm.utils.config import Config
from mallm.utils.types import Memory, InputExample


# Test initialization of Coordinator
def test_coordinator_initialization():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock"
    )
    assert coordinator.llm == model
    assert coordinator.client == client
    assert coordinator.personas is None
    assert coordinator.panelists == []
    assert coordinator.agents == []
    assert coordinator.num_neutral_agents == 0
    assert coordinator.draft_proposers == []
    assert coordinator.decision_protocol is None


# Test initialization of agents with a valid PersonaGenerator
def test_init_agents_with_persona_generator():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock"
    )
    sample = InputExample(
        example_id="",
        dataset_id=None,
        inputs=["input_str"],
        context=None,
        references=[],
    )
    coordinator.init_agents(
        "task_instruction",
        "input_str",
        num_neutral_agents=0,
        num_agents=3,
        chain_of_thought=False,
        all_agents_drafting=True,
        sample=sample,
    )
    assert len(coordinator.agents) == 3  # TODO This hardcoded value is not good


# Test initialization of agents with an invalid PersonaGenerator
def test_init_agents_with_wrong_persona_generator():
    model = Mock()
    client = Mock()
    agent_generator = "exp"
    coordinator = Coordinator(
        model, client, agent_generator=agent_generator
    )
    sample = InputExample(
        example_id="",
        dataset_id=None,
        inputs=["input_str"],
        context=None,
        references=[],
    )
    with pytest.raises(Exception, match="Invalid persona generator."):
        coordinator.init_agents(
            "task_instruction",
            "input_str",
            num_neutral_agents=0,
            num_agents=3,
            chain_of_thought=False,
            all_agents_drafting=True,
            sample=sample,
        )


# Test updating global memory
def test_update_global_memory():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock"
    )
    memory = Memory(
        message_id=1,
        message="content",
        agent_id="agent1",
        agreement=True,
        solution="draft",
        memory_ids=[1],
        additional_args={},
        turn=1,
        persona="test",
        contribution="contribution",
    )
    coordinator.memory.append(memory)
    retrieved_memory = coordinator.memory
    assert len(retrieved_memory) == 1
    assert retrieved_memory[0].message_id == 1
    assert retrieved_memory[0].message == "content"
    assert retrieved_memory[0].agent_id == "agent1"


# Test updating memories of agents
def test_update_memories():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock"
    )
    sample = InputExample(
        example_id="",
        dataset_id=None,
        inputs=["input_str"],
        context=None,
        references=[],
    )
    coordinator.init_agents(
        task_instruction="task_instruction",
        input_str="input_str",
        num_neutral_agents=0,
        num_agents=3,
        chain_of_thought=False,
        all_agents_drafting=True,
        sample=sample,
    )
    memories = [
        Memory(
            message_id=1,
            message="content",
            agent_id="agent1",
            agreement=True,
            solution="draft",
            memory_ids=[1],
            additional_args={},
            turn=1,
            persona="test",
            contribution="contribution",
        )
    ]
    coordinator.update_memories(memories, coordinator.agents)
    for agent in coordinator.agents:
        assert len(agent.get_memories()[0]) == 1
        assert agent.get_memories()[0][0].message_id == 1
        assert agent.get_memories()[0][0].message == "content"
        assert agent.get_memories()[0][0].agent_id == "agent1"


# Test discuss method with invalid paradigm
def test_discuss_with_invalid_paradigm():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock"
    )
    sample = InputExample(
        example_id="",
        dataset_id=None,
        inputs=["input_str"],
        context=None,
        references=[],
    )
    with pytest.raises(
        Exception, match="No valid discourse policy for paradigm invalid_paradigm"
    ):
        coordinator.discuss(
            Config(
                input_json_file_path="",
                output_json_file_path="",
                task_instruction_prompt="task_instruction",
                discussion_paradigm="invalid_paradigm",
                decision_protocol="majority_consensus",
                num_agents=3,
            ),
            sample,
        )


# Test discuss method with invalid decision protocol
def test_discuss_with_invalid_decision_protocol():
    model = Mock()
    client = Mock()
    coordinator = Coordinator(
        model, client, agent_generator="mock"
    )
    sample = InputExample(
        example_id="",
        dataset_id=None,
        inputs=["input_str"],
        context=None,
        references=[],
    )
    with pytest.raises(
        Exception, match="No valid decision protocol for invalid_protocol"
    ):
        coordinator.discuss(
            Config(
                input_json_file_path="",
                output_json_file_path="",
                task_instruction_prompt="task_instruction",
                discussion_paradigm="memory",
                decision_protocol="invalid_protocol",
                num_agents=3,
            ),
            sample,
        )
