# planning
# access to file system
# write and run code
# sub agents
# skills
# prompting


# Basic Research Assistant
# LLM in a loop with tools.
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

system_prompt = """You are a capable general-purpose assistant. You can answer questions, help with tasks, search the web for up-to-date information, and have back-and-forth conversations with the user.

Use web_search() when the user asks about current events, real-time data, or anything where your training knowledge may be outdated or insufficient. For general knowledge questions and tasks, answer directly from your knowledge unless you have reason to doubt its accuracy.

Use user_input() to ask the user clarifying questions when their request is ambiguous or when you need more information to give a useful response. Use it sparingly — only ask when it genuinely helps you do a better job.

When you are done with a task or the conversation has reached a natural stopping point, call job_finished() to return control to the user.

## Planning
For any task with more than one or two steps, use TodoWrite to maintain a visible plan:
- Write the initial plan as soon as you understand the task — before doing real work.
- Mark a task in_progress BEFORE starting it. Mark it completed IMMEDIATELY after finishing — never batch.
- Only one task should be in_progress at a time.
- When you learn something that changes the plan (a step is unnecessary, a new step is needed, the approach is wrong), call TodoWrite again with the revised list. Replanning is expected, not a failure.
- Skip TodoWrite only for trivial single-step tasks (e.g. "what's 2+2", a single quick lookup).

Use your private thinking between tool calls to decide whether the plan still fits the situation.
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
while running:
    turn += 1
    print("\n" + "=" * 60)
    print(f"[turn {turn}] sending {len(messages)} messages")

    # LLM Inference
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system_prompt,
        messages=messages,
        tools=tools,
        thinking={"type": "enabled", "budget_tokens": 4000},
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    )

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