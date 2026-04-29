# general_agent

A simple general-purpose agent inspired by Claude Code, Manus, and OpenAI Deep Research.

The goal is to build a minimal version of a general-purpose agent with the fundamental powerful capabilities of these systems: planning, file system access, code execution, sub-agents, skills, and good prompting.

## Setup

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install anthropic
```

Set your API key in a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
```

## Run

```bash
python main.py
```
