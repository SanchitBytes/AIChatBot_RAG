import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import settings


class Storage:

    def __init__(self):
        self.conn = sqlite3.connect(
            database=settings.DB_PATH,
            check_same_thread=False
        )

        self.checkpointer = SqliteSaver(self.conn)

    def get_checkpointer(self):
        return self.checkpointer

    def list_threads(self):
        threads = []
        for cp in self.checkpointer.list(None):
            tid = cp.config["configurable"]["thread_id"]
            if tid not in threads:
                threads.append(tid)
        return threads

    def get_thread_title(self, chatbot, thread_id: str) -> str:
        """Return first user message as chat title."""
        try:
            state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", [])
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    text = str(msg.content)
                    if len(text) > 30:
                        return text[:30] + "..."
                    return text
        except Exception:
            pass
        return "New Chat"
