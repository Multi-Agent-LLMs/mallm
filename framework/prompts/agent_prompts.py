def create_draft(persona):
    prompt = f'''When faced with a task, you are also provided with the preliminary discussion in the format of a python dictionary. Considering your assigned role, generate a continuation of the dialogue that outlines feedback on the current draft.
Your role is: {persona}
    
Here are some examples:
---
Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
Input: 6 12 1 1


Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the "1 - 1" to "1 + 1".
AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

Finish collaboration!

Final answer: 6 * (1 + 1) + 12 = 24

---
Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word "CHATGPT"; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

Profiles:
    - AI Assistant (you): A super-intelligent AI assistant capable of performing tasks more effectively than humans.
    - Poet: A person who studies and creates poetry. The poet is familiar with the rules and formats of poetry and can provide guidance on how to write a poem.
    - Computer Scientist: A scholar who specializes in the academic study of computer science. The computer scientist is familiar with the concept of a quantum computer and can provide guidance on how to explain it.
    - Ten year old child: A child with a limited English vocabulary and little knowledge about complicated concepts, such as a quantum computer.

Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
Computational wonder of our age,
Harnessing the quantum world's strange ways,
Atoms dance, entwined in dual state,
Tapping secrets hidden in their haze.

Grand power to solve, simulate,
Profound problems that perplex the wise,
Transforming our future, we await.

Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
Computer Scientist: Everything looks good to me!
Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
Computer Scientist: Looking good!
Ten year old child: I like this version a lot!

Finish collaboration!

Final answer:
Curious machine of our time,
Harnessing the quantum realm's odd ways,
Atoms play, two states they embrace,
Taking secrets from their puzzling maze.

Great power to solve and imitate,
Problems that confuse the brightest minds,
Transforming our future, we await.

---
Now, identify the participants, provide their profiles, and collaboratively solve the following task step by step. Remember to present your final solution with the prefix "Final answer:".

Task: Write a short and coherent story about that incorporates the answers to the following questions: 
'''

def create_draft(task_name, task_description, agent_memory, persona, source_text, use_moderator = True):
    if use_moderator:
        prompt = f"You are one of several participants of a turn-based discourse with the goal to solve the {task_name} task. \
            {task_description}. \
            You are the moderator in this discussion. \
            You are required to be objective and non-opinionated. \
            The other participants will provide opinionated feedback on the current draft to solve the {task_name} task. \
            You must try to incorporate everyones feedback as objectively as possible into a new draft that should satify everyones preferences to a maximal degree. \
            The concrete example to solve is: '{source_text}' \
            There has been preliminary discussion about a possible solution for the task. It is provided to you in JSON format: \
            {agent_memory} \
            In this discussion you are now required to create a neutral and objective draft that solves the provided task. \
            Please objectively consider the provided discussion history and other participants recent feedback when creating the updated draft. \
            Answer only with the draft as plain text (no JSON). Do not provide additional text in your response."
    else:
        prompt = f"You are one of several participants of a turn-based discourse with the goal to solve the {task_name} task. \
            {task_description}. \
            Your role is: {persona}. \
            The concrete example to solve is: '{source_text}' \
            There has been preliminary discussion about a possible solution for the task. It is provided to you in JSON format: \
            {agent_memory} \
            In this discussion you are now required to create a draft that solves the provided task and aligns with your beliefs. \
            Please also consider the provided discussion history and other participants recent feedback when creating the updated draft. \
            Answer only with the draft as plain text (no JSON). Do not provide additional text in your response."
    return prompt

def generate_feedback(task_name, task_description, agent_memory, persona, current_draft, source_text, avg_feedback_length = 3):
    prompt = f"You are one of several participants of a turn-based discourse with the goal to solve the {task_name} task. \
        {task_description}. \
        The concrete example to solve is: '{source_text}' \
        There has been preliminary discussion about a possible solution for the task. It is provided to you in JSON format: \
        {agent_memory} \
        The current draft you have to assess is: \
        {current_draft} \
        Your role is: {persona}. \
        In this discussion you are now required to generate feedback on the newest draft as a solution to the task. \
        Make sure to tackle the problem from you point of view having the {persona} role. \
        If you believe that the current draft is fine, you can say that no further changes are needed. \
        Answer only with the feedback within {avg_feedback_length} sentences. Do not provide additional text in your response."
    return prompt

def brainstorm(task_name, task_description, persona, source_text, avg_feedback_length = 3):
    prompt = f"You are one of several participants of a turn-based discourse with the goal to solve the {task_name} task. \
        {task_description}. \
        Your role is: {persona}. \
        The concrete example to solve is: '{source_text}' \
        In this discussion you are now first required to brainstorm about the best possible solution to the task. \
        Answer only with the brainstorming text within {avg_feedback_length} sentences. Do not provide additional text in your response."
    return prompt

def decide_boolean(task_name, task_description, agent_memory, current_draft, feedback, source_text):
    prompt = f"You are one of several participants of a turn-based discourse with the goal to solve the {task_name} task. \
        {task_description}. \
        The concrete example to solve is: '{source_text}' \
        There has been preliminary discussion about a possible solution for the task. It is provided to you in JSON format: \
        {agent_memory} \
        The current draft needs to solve the task as best as possible: \
        {current_draft} \
        This is the feedback you need to assess: {feedback} \
        Does this feedback agree to the draft (AGREE) or does it indicate that further changes are needed (DISAGREE)? Indicate this by answering 'AGREE' or 'DISAGREE'. Do not provide additional text in your response."
    return prompt