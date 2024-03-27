# MALLM
Multi-Agent LLMs For Conversational Task-Solving: Framework

## Install
Create an environment with
`conda create --name mallm --file requirements.txt`
You also need the checkpoints for the LLM you want to use. Currently, LLaMA-2-70b-chat has been tested and is working.

## Project Structure
MALLM is composed of three parts: 
The framework follows this structure and can be found in the `framework` directory.

1) Agents (subdirectory: `framework/agents/`)
2) Discourse Policy (subdirectory: `framework/discourse_policy/`)
3) Decision Making (subdirectory: `framework/decision_making/`)

Experiments can be implemented in the `experiments` directory. Test stuff in the `notebooks` directory.

Please do not develop on master and create a branch. Thank you!
