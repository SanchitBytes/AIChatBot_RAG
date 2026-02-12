from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_chatbot(llm, tools, storage):
    llm_with_tools = llm.bind_tools(tools)

    def chat_node(state: ChatState, config=None):
        thread_id = None
        if config:
            thread_id = config.get("configurable", {}).get("thread_id")
        system = SystemMessage(
            content=(
                "Use rag_tool for PDF questions. "
                f"Thread: {thread_id}"
            )
        )
        messages = [system, *state["messages"]]
        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=storage.get_checkpointer())
