import os
import tempfile
from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RetrieverStore:

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._retrievers: Dict[str, Any] = {}
        self._metadata: Dict[str, dict] = {}

    def get(self, thread_id: str):
        return self._retrievers.get(str(thread_id))

    def ingest(self, file_bytes: bytes, thread_id: str, filename=None):

        if not file_bytes:
            raise ValueError("Empty file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_bytes)
            path = f.name

        try:
            docs = PyPDFLoader(path).load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            store = FAISS.from_documents(chunks, self.embeddings)

            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            self._retrievers[str(thread_id)] = retriever

            meta = {
                "filename": filename or os.path.basename(path),
                "documents": len(docs),
                "chunks": len(chunks),
            }

            self._metadata[str(thread_id)] = meta

            return meta

        finally:
            os.remove(path)

    def has_document(self, thread_id: str):
        return str(thread_id) in self._retrievers

    def get_metadata(self, thread_id: str):
        return self._metadata.get(str(thread_id), {})
