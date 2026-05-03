# planning
# access to file system
# write and run code
# sub agents
# skills
# prompting


# Basic Research Assistant
# LLM in a loop with tools.
import json
import os
import subprocess
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

WORKING_DIR = os.getcwd()
BASH_TIMEOUT = 120
BASH_OUTPUT_CAP = 20_000

system_prompt = """You are a capable general-purpose assistant. You can answer questions, help with tasks, search the web for up-to-date information, and have back-and-forth conversations with the user.

## Planning (MOST IMPORTANT)
For any task with more than one or two steps, your VERY FIRST action MUST be to call TodoWrite with an initial plan — before any web_search, before any other work. This is non-negotiable.

Then as you work:
- Mark a task in_progress BEFORE starting it. Mark it completed IMMEDIATELY after finishing — never batch.
- Only one task should be in_progress at a time.
- When you learn something that changes the plan (a step is unnecessary, a new step is needed, the approach is wrong), call TodoWrite again with the revised list. Replanning is expected, not a failure.
- Skip TodoWrite only for trivial single-step tasks (e.g. "what's 2+2", a single quick lookup, a direct factual question).

Use your private thinking between tool calls to decide whether the plan still fits the situation.

## Other tools
Use web_search() when the user asks about current events, real-time data, or anything where your training knowledge may be outdated or insufficient. For general knowledge questions and tasks, answer directly from your knowledge unless you have reason to doubt its accuracy.

Use user_input() to ask the user clarifying questions when their request is ambiguous or when you need more information to give a useful response. Use it sparingly — only ask when it genuinely helps you do a better job.

Use run_bash() to execute shell commands on the user's machine. Important properties of this tool:
- Every call requires the user to MANUALLY APPROVE the exact command before it runs. You cannot run anything without their explicit consent. If the user denies a command, do NOT immediately retry the same one — adjust your approach or ask them what they'd prefer.
- Each call spawns a fresh, INDEPENDENT shell. State does NOT persist between calls: cwd resets, environment variables are lost, activated virtualenvs deactivate, Python REPL state is gone. Files written to disk and packages installed do persist.
- If you need state within a single operation, chain commands with && in one call, e.g. `cd /path && source myenv/bin/activate && python script.py`.
- Be deliberate. Prefer one well-formed command that accomplishes the goal over many small commands the user has to approve one-by-one.
- Output is truncated at ~20KB and there is a 120s timeout per command.

When you are done with a task or the conversation has reached a natural stopping point, call job_finished() to return control to the user.
"""

def read_multiline(label="User"):
    """Read multi-line input from the user, terminated by an empty line."""
    print(f"{label}: (end with an empty line)\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

prompt = read_multiline("Task")

# Set up conversation
messages = [
    {"role": "user", "content": prompt}
]

# Define tools for LLM to use
tools = [
    {
        "type": "web_search_20260209",
        "name": "web_search"
    },
    {
        "name": "user_input",
        "description": "get input from the user",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "message or question to the user"
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "job_finished",
        "description": "use this when you are finished",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_bash",
        "description": (
            "Run a shell command on the user's machine. The user MUST manually approve every command before it runs — "
            "if denied, the command is not executed and a denial message is returned. "
            "Each call uses a fresh, independent shell: cwd, env vars, and activated virtualenvs do NOT persist between calls. "
            "Chain commands with && in a single call when you need state to carry through (e.g. 'cd dir && source venv/bin/activate && python x.py'). "
            "Output is capped at ~20KB and execution times out at 120s."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run."
                }
            },
            "required": ["command"]
        }
    },
    {
      "name": "TodoWrite",
      "description": "Create or update the structured task plan. Pass the FULL updated list each call.",
      "input_schema": {
        "type": "object",
        "required": ["todos"],
        "additionalProperties": False,
        "properties": {
          "todos": {
            "type": "array",
            "description": "The updated todo list",
            "items": {
              "type": "object",
              "required": ["content", "status", "id"],
              "additionalProperties": False,
              "properties": {
                "content": { "type": "string", "minLength": 1 },
                "status": { "enum": ["pending", "in_progress", "completed"] },
                "id": { "type": "string" }
              }
            }
          }
        }
      }
    }
]

def user_input(message):
    """Prompt the user in the terminal and return their reply."""
    print(message)
    return read_multiline("User")

def job_finished():
    """Signal the main loop to stop."""
    global running
    running = False
    return ""

def run_bash(command):
    """Run a shell command after explicit user approval; returns exit code and (stdout+stderr)."""
    print(f"\n[bash request] cwd={WORKING_DIR}")
    print(f"  $ {command}")
    approval = input("Approve? (y/N): ").strip().lower()
    if approval != "y":
        return "[denied by user] command was NOT executed. Adjust your approach or ask the user what they'd prefer."
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKING_DIR,
            capture_output=True, text=True, timeout=BASH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout after {BASH_TIMEOUT}s] command was killed"
    out = r.stdout or ""
    if r.stderr:
        out += ("\n[stderr]\n" + r.stderr)
    if len(out) > BASH_OUTPUT_CAP:
        out = out[:BASH_OUTPUT_CAP] + f"\n... [+{len(out)-BASH_OUTPUT_CAP} chars truncated]"
    return f"[exit {r.returncode}]\n{out}"

todos = []

def print_todos():
    """Render the current plan to the terminal."""
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    print("\n--- Plan ---")
    for t in todos:
        print(f"{marks.get(t['status'], '[?]')} {t['content']}")
    print("------------\n")

def todo_write(new_todos):
    """Replace the full todo list and pretty-print it."""
    global todos
    todos = new_todos
    print_todos()
    return new_todos


def call_function(name, args):
    """Dispatch a tool call by name to its Python handler."""
    if name == "user_input":
        return user_input(**args)
    elif name == "job_finished":
        return job_finished(**args)
    elif name == "TodoWrite":
        return todo_write(args["todos"])
    elif name == "run_bash":
        return run_bash(**args)

def truncate(s, n=600):
    """Truncate long strings for readable logs."""
    s = s if isinstance(s, str) else json.dumps(s, default=str)
    return s if len(s) <= n else s[:n] + f"... [+{len(s)-n} chars]"

def log_block(block):
    """Print one response block with a clear marker for dev visibility."""
    t = block.type
    if t == "text":
        print(f"\n[text]\n{block.text}")
    elif t == "thinking":
        print(f"\n[thinking]\n{truncate(block.thinking, 1200)}")
    elif t == "tool_use":
        print(f"\n[tool_use] {block.name}({truncate(block.input)})")
    elif t == "server_tool_use":
        print(f"\n[server_tool_use] {block.name}({truncate(block.input)})")
    elif t == "web_search_tool_result":
        results = getattr(block, "content", [])
        if isinstance(results, list):
            print(f"\n[web_search_results] {len(results)} hits")
            for i, r in enumerate(results[:5], 1):
                print(f"  {i}. {getattr(r, 'title', '?')} — {getattr(r, 'url', '?')}")
        else:
            print(f"\n[web_search_results] {truncate(results)}")
    else:
        print(f"\n[{t}] {truncate(getattr(block, 'model_dump', lambda: block)())}")

running = True
turn = 0
container_id = None
while running:
    turn += 1
    print("\n" + "=" * 60)
    print(f"[turn {turn}] sending {len(messages)} messages" + (f" | container={container_id}" if container_id else ""))

    # LLM Inference
    kwargs = dict(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system_prompt,
        messages=messages,
        tools=tools,
        thinking={"type": "enabled", "budget_tokens": 4000},
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    )
    if container_id:
        kwargs["container"] = container_id
    response = client.messages.create(**kwargs)

    if response.container:
        container_id = response.container.id

    u = response.usage
    print(f"[stop_reason] {response.stop_reason} | [usage] in={u.input_tokens} out={u.output_tokens}")

    # Add assistant response to conversation history
    messages.append({"role": "assistant", "content": response.content})

    # Calling tools
    tool_results = []
    for block in response.content:
        log_block(block)
        if block.type == "tool_use":
            result = call_function(block.name, block.input)
            print(f"[tool_result] {truncate(result)}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result)
            })

    if tool_results and running:
        messages.append({"role": "user", "content": tool_results})
    elif response.stop_reason == "end_turn":
        # Agent ended its turn without a tool call. Let the user reply.
        follow_up = read_multiline("User")
        if not follow_up.strip():
            break
        messages.append({"role": "user", "content": follow_up})





# --> actual basic agent.
# --> check out Shell/Computer use