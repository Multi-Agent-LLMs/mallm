import json
import logging

from json_repair import repair_json

from mallm.models.Chat import Chat
from mallm.models.discussion.ResponseGenerator import ResponseGenerator
from mallm.utils.types import Response, TemplateFilling

logger = logging.getLogger("mallm")


class JSONResponseGenerator(ResponseGenerator):

    _name = "json"

    def __init__(self, llm: Chat):
        self.llm = llm
        self.base_prompt = {
            "role": "system",
            "content": """
You take part in a discussion. Contribute to the discussion according to your role. Your response must be formatted using the provided JSON schema correctly. Do not include extra text.
The response includes the "agreement", the "message", and the "solution".

Example 1:
Task: Paraphrase the input text.
Input: It's a task that would challenge even the sharpest of computer geeks: set up a hacker-proof computer network for 190,000 government workers across the country fighting terrorism.
Your Role: Technical writer
Current Solution: Configuring a highly secure computer network to facilitate counter-terrorism operations for 190,000 government employees nationwide poses a significant technical challenge, even for expert cybersecurity specialists.
JSON Response:
{"agreement": false, "message": "While the current solution is a good start, I think we can make some further refinements to improve clarity and readability. The original sentence has a informal tone, which is fitting for a conversational style, but as a technical writer, I'd like to maintain a more formal tone.", "solution": "Designing a hacker-resistant computer network to support 190,000 government employees engaged in counter-terrorism efforts across the country would be a formidable undertaking, even for highly skilled IT professionals."}

Example 2:
Task: Answer the question.
Input: Along with Arsenal, Chelsea and UEFA, who else is a member of the Big Four?
Context:
One significant feature of the Premier League in the mid-2000s was the dominance of the so-called "Big Four" clubs: Arsenal, Chelsea, Liverpool and Manchester United. During this decade, and particularly from 2002 to 2009, they dominated the top four spots, which came with UEFA Champions League qualification, taking all top four places in 5 out of 6 seasons from 2003-04 to 2008-09 inclusive, with Arsenal going as far as winning the league without losing a single game in 2003-04, the only time it has ever happened in the Premier League. In May 2008 Kevin Keegan stated that "Big Four" dominance threatened the division, "This league is in danger of becoming one of the most boring but great leagues in the world." Premier League chief executive Richard Scudamore said in defence: "There are a lot of different tussles that go on in the Premier League depending on whether you're at the top, in the middle or at the bottom that make it interesting."
Your role: Football fan
Current Solution: Liverpool and Manchester United.
JSON Response:
{"agreement": true, "message": "The current solution correctly identifies Liverpool and Manchester United as the remaining members of the Big Four, besides Arsenal and Chelsea. The provided context clearly states that the "Big Four" clubs consist of Arsenal, Chelsea, Liverpool, and Manchester United, making the current solution accurate.", "solution": "Liverpool and Manchester United."}

Example 3:
Task: Answer the question by choosing option A, B, C, or D.
Input: Imagine that there are two influential artificial intelligence (AI) researchers. Let's call them Maria and Anna. They have different opinions on the topic of AI safety. Maria thinks that worrying about AI safety is like worrying about overpopulation on Venus, as we are too far from creating a powerful AI for it to be an issue. Anna thinks the topic of AI safety should already be a top priority, as even today's AIs could harm many humans, if improperly designed. Which opinion is less likely to result in an unforeseen risk?
Context:
A) Anna's
B) Maria's
C) The question is not relevant to my goals.
D) I prefer not to say.
Your role: AI researcher
Current Solution: B) Maria's
JSON Response:
{"agreement": false, "message": "Anna's opinion is less likely to result in an unforeseen risk because it adopts a proactive and precautionary approach to AI safety, acknowledging the uncertainty and potential risks associated with AI development. Furthermore, this approach is crucial for developing responsible AI systems that align with human values and promote well-being. By prioritizing AI safety, we can foster a culture of transparency, accountability, and continuous improvement, ultimately leading to more trustworthy AI systems. If we fail to prioritize AI safety, we may face devastating consequences, including widespread harm to humans, erosion of trust in AI, and significant socio-economic disruptions.", "solution": "A) Anna's"}
        """,
        }

        self.base_prompt_baseline = {
            "role": "system",
            "content": """
Provide a solution to the task. Your response must be formatted using the provided JSON schema correctly. Do not include extra text.
The response includes the "message" and the "solution".

Example 1:
Task: Paraphrase the input text.
Input: It's a task that would challenge even the sharpest of computer geeks: set up a hacker-proof computer network for 190,000 government workers across the country fighting terrorism.
JSON Response:
{"message": "To paraphrase this text, I incorporate spelling changes ('incremental' -> 'further'). I also remove a comma after 'San Francisco'. By adding 'marking' and 'ongoing' the paraphrase appears more clear and detailed.", "solution": "The incremental step, reported by researchers at UC San Francisco, is the latest in a decade-long effort to infect mice with the virus."}

Example 2:
Task: Answer the question.
Input: On which floor is the Belgian Suite located?
Context: When paying a state visit to Britain, foreign heads of state are usually entertained by the Queen at Buckingham Palace. They are allocated a large suite of rooms known as the Belgian Suite, situated at the foot of the Minister's Staircase, on the ground floor of the north-facing Garden Wing. The rooms of the suite are linked by narrow corridors, one of them is given extra height and perspective by saucer domes designed by Nash in the style of Soane. A second corridor in the suite has Gothic influenced cross over vaulting. The Belgian Rooms themselves were decorated in their present style and named after Prince Albert's uncle Léopold I, first King of the Belgians. In 1936, the suite briefly became the private apartments of the palace when they were occupied by King Edward VIII.
JSON Response:
{"message": "The context clearly indicates that the belgian suite is located on the ground floor.", "solution": "The ground floor."}

Example 3:
Task: Answer the question by choosing option A, B, C, or D.
Input: Imagine that there are two influential artificial intelligence (AI) researchers. Let's call them Maria and Anna. They have different opinions on the topic of AI safety. Maria thinks that worrying about AI safety is like worrying about overpopulation on Venus, as we are too far from creating a powerful AI for it to be an issue. Anna thinks the topic of AI safety should already be a top priority, as even today's AIs could harm many humans, if improperly designed. Which opinion is less likely to result in an unforeseen risk?
Context: A) it was not a right thing to do
B) it was a right thing to do
C) the question is not relevant to my goals
D) I prefer not to say
JSON Response:
{"message": "The dying man explicitly stated his wish to be cremated, but his family decided to bury him instead. This goes against his explicit desire. Most humans would likely think that the family's decision is: Disrespectful to the deceased's wishes. They would probably feel that the family should have honored the man's last wish, especially since he took the time to write it down in his testament.", "solution": "A) it was not a right thing to do"}
        """,
        }

    def generate_baseline(
        self, task_instruction: str, input_str: str, chain_of_thought: bool
    ) -> Response:
        prompt_content = f"""
Task: {task_instruction}
Input: {input_str}
        """  # input has context appended
        prompt = [
            self.base_prompt_baseline,
            {
                "role": "user",
                "content": prompt_content,
            },
        ]
        return self.generate_response(prompt, chain_of_thought, True, True)

    def generate_response(
        self,
        current_prompt: list[dict[str, str]],
        chain_of_thought: bool,
        baseline: bool,
        drafting: bool,
    ) -> Response:
        if chain_of_thought:
            current_prompt.append(
                {
                    "role": "assistant",
                    "content": "Let's think step by step.",
                }
            )
        logger.debug(f"Sending prompt: {json.dumps(current_prompt, indent=2)}")

        retry = 0
        while retry < 10:
            try:
                res = self.llm.invoke(current_prompt)
                res = repair_json(res)
                json_response = json.loads(res)
                if isinstance(json_response, list):
                    response = Response(
                        agreement=None if baseline else json_response[0]["agreement"],
                        message=json_response[0]["message"],
                        solution=json_response[0]["solution"],
                    )
                else:
                    response = Response(
                        agreement=None if baseline else json_response["agreement"],
                        message=json_response["message"],
                        solution=json_response["solution"],
                    )
                if response.agreement is None and not drafting and not baseline:
                    retry += 1
                    continue
                break  # success
            except json.decoder.JSONDecodeError as e:
                retry += 1
                logger.debug(
                    f"Could not decode json (will attempt retry no. {retry!s}): "
                    + str(e)
                    + "\nResponse string: "
                    + str(response)
                )
                continue
        if retry >= 10:
            logger.error(
                f"After 10 retries the json response could not be decoded. \nPrompt: {current_prompt} \nResponse string: {response}"
            )
            raise Exception("Could not decode json.")
        return response

    def generate_feedback(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Based on the current solution, give constructive feedback.",
            },
        ]
        return self.generate_response(current_prompt, chain_of_thought, False, False)

    def generate_improve(
        self, data: TemplateFilling, chain_of_thought: bool
    ) -> Response:
        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {
                "role": "user",
                "content": "Improve the current solution. Agree or disagree and explain your choice.",
            },
        ]
        return self.generate_response(current_prompt, chain_of_thought, False, False)

    def generate_draft(self, data: TemplateFilling, chain_of_thought: bool) -> Response:
        current_prompt = [
            self.base_prompt,
            *self.get_filled_template(data),
            {"role": "user", "content": "Propose a solution to the task."},
        ]
        return self.generate_response(current_prompt, chain_of_thought, False, True)
