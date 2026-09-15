import os
import streamlit as st
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from duckduckgo_search import DDGS

# Page Config
st.set_page_config(page_title="LangGraph ReAct Agent", page_icon="🤖")
st.title("🤖 LangGraph ReAct Agent UI")

# Sidebar for API Key Setup
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Anthropic API Key", type="password")
if api_key:
    os.environ["ANTHROPIC_API_KEY"] = api_key

# Define Tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * (4 + 3)'."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression) <= allowed_chars:
            return "Error: Invalid characters."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
    except Exception as e:
        return f"Search failed: {e}"

tools = [calculator, web_search]

# Define State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Graph Compilation Helper
@st.cache_resource
def get_graph():
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    SYSTEM_PROMPT = "You are a helpful assistant. Use tools when needed."

    def agent_node(state: AgentState):
        messages = state["messages"]
        if not any(getattr(m, "type", "") == "system" for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()

# Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# User Input Box
if user_query := st.chat_input("Ask something..."):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Please enter your API Key in the sidebar!")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.write(user_query)

        app = get_graph()
        with st.spinner("Thinking & Executing Tools..."):
            result = app.invoke({"messages": st.session_state.messages})
            st.session_state.messages = result["messages"]
            final_answer = st.session_state.messages[-1].content

        with st.chat_message("assistant"):
            st.write(final_answer)