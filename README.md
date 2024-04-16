<br />
<p align="center">
<a><img src="image/mallm.webp" alt="MALLM" width="128" height="128" title="FawnRescue"></a>
  <h3 align="center">MALLM</h3>
  <p align="center">
    Multi-Agent LLMs For Conversational Task-Solving: Framework<br />
    <p align="center">
  <a href="https://github.com/jonas-becker/mallm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/FawnRescue/drone" alt="License"></a>
  <a href="https://github.com/jonas-becker/mallm/network/members"><img src="https://img.shields.io/github/forks/FawnRescue/drone?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/jonas-becker/mallm/stargazers"><img src="https://img.shields.io/github/stars/FawnRescue/drone?style=social" alt="GitHub stars"></a>
</p>
    <p>
    <a href="https://github.com/jonas-becker/mallm/issues">Report Bug</a>
    ·
    <a href="https://github.com/jonas-becker/mallm/issues">Request Feature</a>
    </p>
  </p>
</p>


## Install
Create an environment with
`conda env create -f environment.yml`

### Package
Install as a package
`pip install -e .`

### Test Data
Create the test data
`python experiments/data/data_downloader.py`

You also need the checkpoints for the LLM you want to use. Currently, LLaMA-2-70b-chat has been tested and is working.

## Project Structure
MALLM is composed of three parts: 
The framework follows this structure and can be found in the `framework` directory.

1) Agents (subdirectory: `framework/agents/`)
2) Discourse Policy (subdirectory: `framework/discourse_policy/`)
3) Decision Making (subdirectory: `framework/decision_making/`)

Experiments can be implemented in the `experiments` directory. Test stuff in the `notebooks` directory.

Please do not develop on master and create a branch. Thank you!
