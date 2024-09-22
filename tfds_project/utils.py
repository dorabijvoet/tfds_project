import io
import os
from PIL import Image


def remove_duplicates_keep_highest_score(documents):
    unique_docs = {}

    for doc in documents:
        doc_id = doc.metadata.get("doc_id")
        if doc_id in unique_docs:
            if (
                doc.metadata["reranking_score"]
                > unique_docs[doc_id].metadata["reranking_score"]
            ):
                unique_docs[doc_id] = doc
        else:
            unique_docs[doc_id] = doc

    return list(unique_docs.values())
