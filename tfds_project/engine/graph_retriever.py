from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from typing import List


class GraphRetriever(BaseRetriever):
    vectorstore: VectorStore
    sources: list = ["OWID"]
    threshold: float = 0.5
    k_total: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # Check if all elements in the list are IEA or OWID
        assert isinstance(self.sources, list)
        assert self.sources
        assert any([x in ["OWID"] for x in self.sources])

        docs = self.vectorstore.similarity_search_with_score(
            query=query, k=self.k_total
        )

        # Filter if scores are below threshold
        # docs = [x for x in docs if x[1] > self.threshold]

        # Remove duplicate documents
        unique_docs = []
        seen_docs = []
        for i, doc in enumerate(docs):
            if doc[0].page_content not in seen_docs:
                unique_docs.append(doc)
                seen_docs.append(doc[0].page_content)

        # Add score to metadata
        results = []
        for i, (doc, score) in enumerate(unique_docs):
            doc.metadata["similarity_score"] = score
            doc.metadata["content"] = doc.page_content
            results.append(doc)

        return results
