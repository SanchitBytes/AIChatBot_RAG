import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from core.graph import create_chatbot
from core.llm import create_embeddings, create_llm
from core.retriever import RetrieverStore
from core.storage import Storage
from core.tools import create_tools

# ================= Bootstrap =================

@st.cache_resource
def bootstrap():

    llm = create_llm()
    embeddings = create_embeddings()

    retrievers = RetrieverStore(embeddings)
    storage = Storage()

    tools = create_tools(retrievers)

    chatbot = create_chatbot(llm, tools, storage)

    return chatbot, retrievers, storage


chatbot, retrievers, storage = bootstrap()


# ================= Utilities =================

def new_thread():
    return str(uuid.uuid4())


def reset_chat():
    new_id = new_thread()

    st.session_state.thread_id = new_id
    st.session_state.messages = []

    # Refresh threads from DB
    st.session_state.threads = storage.list_threads()
    st.session_state.thread_titles = {}


# ================= Session Init =================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "threads" not in st.session_state:
    st.session_state.threads = storage.list_threads()

if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}


# ================= Sidebar =================

st.sidebar.title("PDF Chatbot")

# New Chat
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Upload PDF
uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded:
    meta = retrievers.ingest(
        uploaded.getvalue(),
        st.session_state.thread_id,
        uploaded.name
    )
    st.sidebar.success(f"Indexed: {meta['filename']}")

# Past Conversations
st.sidebar.subheader("Past Conversations")

threads = st.session_state.get("threads", [])
titles = st.session_state.get("thread_titles", {})

if not threads:
    st.sidebar.caption("No previous chats")
else:
    for tid in reversed(threads):

        # Load title if not cached
        if tid not in titles:
            titles[tid] = storage.get_thread_title(chatbot, tid)

        label = titles.get(tid, "New Chat")
        if st.sidebar.button(
            label,
            key=f"thread-{tid}",
            use_container_width=True
        ):
            st.session_state.thread_id = tid
            state = chatbot.get_state(
                config={"configurable": {"thread_id": tid}}
            )
            history = []
            for msg in state.values.get("messages", []):
                role = "user" if msg.type == "human" else "assistant"
                history.append({
                    "role": role,
                    "content": msg.content
                })
            st.session_state.messages = history
            st.rerun()


# ================= Main =================

st.title("Multi Utility Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_input = st.chat_input("Ask something")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant"):

        def stream():
            final_answer = ""
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # Only show normal AI text (not tool calls)
                if isinstance(chunk, AIMessage):
                    # Skip function/tool responses
                    if chunk.content and isinstance(chunk.content, str):
                        final_answer += chunk.content
                        yield chunk.content

            return final_answer

        answer = st.write_stream(stream)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Refresh thread list and titles
    st.session_state.threads = storage.list_threads()
    st.session_state.thread_titles = {}

