# LangGraph ReAct Agent — Agentic AI Assignment

A complete **agentic AI** implementation built with [LangGraph](https://langchain-ai.github.io/langgraph/), submitted as a Jupyter Notebook.

## What makes this "agentic"?

Unlike a plain chatbot that just replies once, this agent:
1. **Reasons** about what it needs to do.
2. **Decides on its own** whether to call a tool (calculator, web search).
3. **Observes** the tool's result.
4. **Loops** back into reasoning until it's ready to give a final answer.

This think → act → observe cycle is the core definition of an "agent" in agentic AI.

## Files in this project

| File | Purpose |
|---|---|
| `LangGraph_Agent.ipynb` | **Main deliverable.** The full agent: setup, tools, state, graph, and example runs — with explanations in markdown cells. |
| `requirements.txt` | Python dependencies (for reference — the notebook also installs them via `!pip install` in its first code cell). |
| `README.md` | This file. |

## How to run

1. Open `LangGraph_Agent.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. Run the first code cell to install dependencies.
3. Add your API key in the "Set your API key" cell:
   ```python
   os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
   ```
   (or switch to OpenAI by uncommenting the `LLM_PROVIDER = "openai"` lines and adding `OPENAI_API_KEY`)
4. Run the remaining cells in order, top to bottom.
5. In the "Try it out" and "Interactive chat" cells, test the agent with your own prompts, e.g.:
   - `"What is 342 * 17?"` → uses the calculator tool
   - `"What's the latest news about SpaceX?"` → uses the web_search tool
   - `"Hi, how are you?"` → answers directly, no tool needed

**Note for reviewers:** an API key (Anthropic or OpenAI) is required to actually execute the LLM calls. Without one, the notebook will run up through the graph-building cells but will error when invoking the agent.

## How the graph works

```
        START
          |
          v
     +----------+
     |  agent   |  <--------------------+
     +----------+                       |
          |                             |
   should_continue?                     |
      /        \\                        |
   tools      END                       |
     |                                  |
     +----------------------------------+
```

- **`agent` node**: the LLM looks at the conversation and decides whether it
  needs a tool, or can answer directly.
- **`should_continue`**: checks if the LLM's last response requested a tool call.
- **`tools` node**: actually executes the requested tool(s) and returns results.
- The result feeds back into `agent`, which reasons again — this is the loop.

## Extending this project

- **Add more tools**: write a new function, decorate it with `@tool`, add it to
  the `tools` list.
- **Multi-agent setup**: create multiple node functions (e.g. a "researcher" and
  a "writer"), each with their own system prompt, and add edges between them.
- **Persistent memory across sessions**: LangGraph supports checkpointing
  (e.g. `MemorySaver`) for remembering past conversations between runs.
- **Streaming output**: use `app.stream(...)` instead of `app.invoke(...)` to
  show intermediate reasoning steps.
