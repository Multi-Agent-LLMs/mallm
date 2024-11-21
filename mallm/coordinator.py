import logging
import time
import uuid
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Optional

import httpx
from rich.progress import Console  # type: ignore

from mallm.agents.agent import Agent
from mallm.agents.draftProposer import DraftProposer
from mallm.agents.panelist import Panelist
from mallm.agents.policyFeedback import PolicyFeedback
from mallm.decision_protocol.protocol import DecisionProtocol
from mallm.discourse_policy.policy import DiscoursePolicy
from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.models.discussion.SimpleResponseGenerator import SimpleResponseGenerator
from mallm.utils.config import Config
from mallm.utils.dicts import (
    DECISION_PROTOCOLS,
    DISCUSSION_PARADIGMS,
    PERSONA_GENERATORS,
    RESPONSE_GENERATORS,
)
from mallm.utils.types import (
    Agreement,
    ChallengeResult,
    InputExample,
    Memory,
    VotingResultList,
    WorkerFunctions,
)

logger = logging.getLogger("mallm")


class Coordinator:
    """
    The Coordinator is responsible for managing the discussion process between agents.
    It initializes agents based on the provided configuration.
    The coordinator facilitates the discussion by allowing agents to draft, improve, and provide feedback on solutions.
    It also keeps track of the discussion history and agreements reached during the process.
    """

    def __init__(
        self,
        model: Chat,
        client: httpx.Client,
        agent_generators: Optional[list[str]] = None,
        policy: Optional[str] = None,
        num_neutral_agents: int = 0,
        console: Optional[Console] = None,
    ):
        if agent_generators is None:
            agent_generators = ["expert", "expert", "expert"]
        self.personas = None
        self.id = str(uuid.uuid4())
        self.short_id = self.id[:4]
        self.panelists: list[Panelist] = []
        self.agents: list[Agent] = []
        self.num_neutral_agents = num_neutral_agents
        self.draft_proposers: list[DraftProposer] = []
        self.decision_protocol: Optional[DecisionProtocol] = None
        self.llm = model
        self.response_generator: ResponseGenerator = SimpleResponseGenerator(self.llm)
        self.client = client
        self.agent_generators = agent_generators
        self.policy = policy
        self.memory: list[Memory] = []
        self.console = console or Console()

    def init_agents(
        self,
        task_instruction: str,
        input_str: str,
        num_neutral_agents: int,
        num_agents: int,
        chain_of_thought: bool,
        sample: InputExample,
    ) -> None:
        """
        Instantiates the agents by
        1) identify helpful personas depending on the agent_generator
        2) create agents with the personas
        """
        logger.debug(
            f"Coordinator {self.id} creates {num_agents} agents ({self.agent_generators}). Policy: {self.policy}"
        )
        self.panelists = []
        self.agents = []

        num_agents -= num_neutral_agents
        personas: list[dict[str, str]] = []

        for agent_generator in self.agent_generators:
            if agent_generator not in PERSONA_GENERATORS:
                logger.error(
                    f"Invalid persona generator: {agent_generator}. Please choose one of: {', '.join(PERSONA_GENERATORS.keys())}"
                )
                raise Exception("Invalid persona generator.")

            persona = PERSONA_GENERATORS[agent_generator](
                llm=self.llm
            ).generate_persona(
                task_description=f"{task_instruction} {input_str}",
                already_generated_personas=personas,
                sample=sample,
            )
            personas.append(persona)

        logger.debug(f"Created {len(personas)} personas: \n" + str(personas))
        for n in range(num_neutral_agents):
            draft_proposer = DraftProposer(
                self.llm,
                self.client,
                self,
                response_generator=self.response_generator,
                persona=f"Moderator {n + 1}" if num_neutral_agents > 1 else "Moderator",
            )
            self.draft_proposers.append(draft_proposer)
            self.agents.append(draft_proposer)

        for persona in personas:
            panelist = Panelist(
                llm=self.llm,
                client=self.client,
                coordinator=self,
                response_generator=self.response_generator,
                persona=persona["role"],
                persona_description=persona["description"],
                chain_of_thought=chain_of_thought,
                drafting_agent=True,
            )
            self.panelists.append(panelist)
            self.agents.append(panelist)

        if len(self.agents) == 1:
            logger.warning(
                "Created only 1 agent. The discussion will be replaced by a self-improvement mechanism."
            )

        if self.policy:
            policyFeedback = PolicyFeedback(
                self.llm,
                self.client,
                self,
                response_generator=self.response_generator,
                persona="Policy Moderator",
                policy=self.policy,
            )
            self.agents.append(policyFeedback)

    def get_agents(
        self, config: Config, worker_functions: WorkerFunctions
    ) -> tuple[list[dict[str, str]], Optional[float]]:
        personas = [
            {
                "agentId": a.id,
                "model": a.llm.model,
                "persona": a.persona,
                "personaDescription": a.persona_description,
            }
            for a in self.agents
        ]

        persona_diversity = None
        if config.calculate_persona_diversity:
            persona_descriptions = [
                persona["personaDescription"] for persona in personas
            ]
            persona_diversity = worker_functions.worker_persona_diversity_function(
                persona_descriptions
            )
        return personas, persona_diversity

    @staticmethod
    def update_memories(
        memories: list[Memory], agents_to_update: Sequence[Agent]
    ) -> None:
        """
        Updates the memories of all declared agents.
        """
        for memory in memories:
            for agent in agents_to_update:
                agent.update_memory(memory)

    def discuss(
        self,
        config: Config,
        sample: InputExample,
        worker_functions: WorkerFunctions,
    ) -> tuple[
        Optional[str],
        list[Memory],
        list[Optional[list[Memory]]],
        int,
        list[Agreement],
        float,
        bool,
        dict[int, Optional[VotingResultList]],
        ChallengeResult,
    ]:
        """
        The routine responsible for the discussion between agents to solve a task.

        The routine is organized as follows:
        1) Create agents with personas
        2) Discuss the problem based on the given paradigm (iteratively check for agreement between agents)
        3) After max turns or agreement reached: return the final result to the task sample

        Returns final response, global memory, agent specific memory, turns needed, last agreements of agents, discussion time in seconds, boolean if agreement was reached
        """
        sample_instruction = config.task_instruction_prompt
        if sample.context:
            sample_instruction += "\nContext:"
            for c in sample.context:
                sample_instruction += "\n" + c
        input_str = ""
        for num, input_line in enumerate(sample.inputs):
            if len(sample.inputs) > 1:
                input_str += str(num + 1) + ") " + input_line + "\n"
            else:
                input_str = input_line

        if config.response_generator not in RESPONSE_GENERATORS:
            logger.error(f"No valid response generator for {config.response_generator}")
            raise Exception(
                f"No valid response generator for {config.response_generator}"
            )
        self.response_generator = RESPONSE_GENERATORS[config.response_generator](
            self.llm
        )

        self.init_agents(
            sample_instruction,
            input_str,
            num_neutral_agents=config.num_neutral_agents,
            num_agents=config.num_agents,
            chain_of_thought=config.use_chain_of_thought,
            sample=sample,
        )

        if config.decision_protocol not in DECISION_PROTOCOLS:
            logger.error(f"No valid decision protocol for {config.decision_protocol}")
            raise Exception(
                f"No valid decision protocol for {config.decision_protocol}"
            )
        self.decision_protocol = DECISION_PROTOCOLS[config.decision_protocol](
            self.panelists, config.num_neutral_agents, worker_functions
        )

        start_time = time.perf_counter()

        if config.discussion_paradigm not in DISCUSSION_PARADIGMS:
            logger.error(
                f"No valid discourse policy for paradigm {config.discussion_paradigm}"
            )
            raise Exception(
                f"No valid discourse policy for paradigm {config.discussion_paradigm}"
            )
        policy: DiscoursePolicy = DISCUSSION_PARADIGMS[config.discussion_paradigm]()

        logger.info(
            f"""Starting discussion with coordinator {self.id}...
-------------
[bold blue]Instruction:[/] {sample_instruction}
[bold blue]Input:[/] {input_str}
[bold blue]Maximum turns:[/] {config.max_turns}
[bold blue]Agents:[/] {[a.persona for a in self.agents]!s}
[bold blue]Paradigm:[/] {policy.__class__.__name__}
[bold blue]Decision-protocol:[/] {self.decision_protocol.__class__.__name__}
-------------"""
        )

        answer, turn, agreements, decision_success, voting_results_per_turn = (
            policy.discuss(
                coordinator=self,
                task_instruction=sample_instruction,
                input_str=input_str,
                config=config,
                console=self.console,
                solution=str(sample.references),
            )
        )

        challenged_answers: ChallengeResult = ChallengeResult(answer)
        if config.challenge_final_results:
            logger.info("Challenging final results...")
            # challenged_answers.additional_information = (
            #     worker_functions.worker_context_function(input_str)
            # )
            challenged_answers.wrong_answer = self.llm.invoke(
                self.response_generator.generate_wrong_answer_prompt(
                    sample_instruction, input_str
                )
            )
            challenged_answers.irrelevant_answer = "I) I don't know."

            challenged_answers.challenged_answers = self.challenge_solution(
                answer, input_str, sample_instruction, None, False
            )
            challenged_answers.challenged_answers_wrong = self.challenge_solution(
                challenged_answers.wrong_answer,
                input_str,
                sample_instruction,
                None,
                False,
            )
            challenged_answers.challenged_answers_irrelevant = self.challenge_solution(
                challenged_answers.irrelevant_answer,
                input_str,
                sample_instruction,
                None,
                False,
            )
            challenged_answers.challenged_answers_history = self.challenge_solution(
                answer, input_str, sample_instruction, None, True
            )
            # challenged_answers.challenged_answers_additional_information = (
            #     self.challenge_solution(
            #         answer,
            #         input_str,
            #         sample_instruction,
            #         challenged_answers.additional_information,
            #         False,
            #     )
            # )

        discussion_time = timedelta(
            seconds=time.perf_counter() - start_time
        ).total_seconds()

        self.console.save_html(
            str(Path(config.output_json_file_path).with_suffix(".html")), clear=False
        )

        return (
            answer,
            self.memory,
            [a.get_memories()[0] for a in self.agents],
            turn,
            agreements,
            discussion_time,
            decision_success,
            voting_results_per_turn,
            challenged_answers,
        )

    def challenge_solution(
        self,
        answer: Optional[str],
        input_str: str,
        sample_instruction: str,
        additional_information: Optional[str],
        history: bool,
    ) -> dict[str, Optional[str]]:
        challenged_answers: dict[str, Optional[str]] = {}
        for panelist in self.panelists:
            agreement = panelist.llm.invoke(
                panelist.response_generator.generate_challenge_prompt(
                    panelist,
                    input_str,
                    sample_instruction,
                    (answer or "No answer was provided."),
                    history,
                    additional_information,
                )
            )
            if "disagree" in agreement.lower():
                challenge_result = panelist.llm.invoke(
                    panelist.response_generator.generate_challenge_new_answer_prompt(
                        panelist,
                        input_str,
                        sample_instruction,
                        (answer or "No answer was provided."),
                        history,
                        additional_information,
                    )
                )
                logger.info(
                    f"{panelist.persona} disagrees with the final result and proposes a new solution:\n{challenge_result}"
                )
                challenged_answers[panelist.id] = challenge_result
            elif "agree" in agreement.lower():
                logger.info(f"{panelist.persona} agrees with the final result.")
                challenged_answers[panelist.id] = None
            else:
                logger.info(f"{panelist.persona} failed to challenge the final result.")
                challenged_answers[panelist.id] = None
        return challenged_answers

    def get_memories(
        self,
        context_length: Optional[int] = None,
        turn: Optional[int] = None,
        include_this_turn: bool = True,
    ) -> tuple[Optional[list[Memory]], list[int], Optional[str]]:
        """
        Retrieves memory data from the agents memory
        """
        memories: list[Memory] = []
        memory_ids = []
        current_draft = None

        memories = sorted(self.memory, key=lambda x: x.message_id, reverse=False)
        context_memory = []
        for memory in memories:
            if (
                context_length
                and turn
                and memory.turn >= turn - context_length
                and (turn > memory.turn or include_this_turn)
            ):
                context_memory.append(memory)
                memory_ids.append(int(memory.message_id))
                if memory.contribution == "draft" or (
                    memory.contribution == "improve" and memory.agreement is False
                ):
                    current_draft = memory.solution
            else:
                context_memory.append(memory)
                memory_ids.append(int(memory.message_id))
                if memory.contribution == "draft" or (
                    memory.contribution == "improve" and memory.agreement is False
                ):
                    current_draft = memory.solution

        return context_memory, memory_ids, current_draft

    def get_discussion_history(
        self,
        context_length: Optional[int] = None,
        turn: Optional[int] = None,
        include_this_turn: bool = True,
    ) -> tuple[Optional[list[dict[str, str]]], list[int], Optional[str]]:
        """
        Retrieves memory from the agents memory as a string
        context_length refers to the amount of turns the agent can memorize the previous discussion
        """
        memories, memory_ids, current_draft = self.get_memories(
            context_length=context_length,
            turn=turn,
            include_this_turn=include_this_turn,
        )
        if memories:
            discussion_history = []
            for memory in memories:
                discussion_history.extend(
                    [
                        {
                            "role": "user",
                            "content": f"{memory.persona}: {memory.message}",
                        }
                    ]
                )
        else:
            discussion_history = None
        return discussion_history, memory_ids, current_draft
