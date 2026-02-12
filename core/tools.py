from typing import Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


def create_tools(retriever_store):

    search_tool = DuckDuckGoSearchRun(region="us-en")

    @tool
    def rag_tool(query: str, thread_id: Optional[str] = None):
        """
        Retrieve relevant information from the uploaded PDF
        associated with the current conversation thread.

        Use this tool when the user asks questions
        related to previously uploaded documents.
        """

        retriever = retriever_store.get(thread_id)

        if retriever is None:
            return {
                "error": "No document indexed. Upload first.",
                "query": query,
            }

        docs = retriever.invoke(query)

        return {
            "query": query,
            "context": [d.page_content for d in docs],
            "metadata": [d.metadata for d in docs],
            "source_file": retriever_store
            .get_metadata(thread_id)
            .get("filename"),
        }

    return [search_tool, rag_tool]
