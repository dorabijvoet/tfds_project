import streamlit as st
import pandas as pd
import numpy as np
import os

# from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from tfds_project.engine.embeddings import get_embeddings_function
from tfds_project.engine.llm import get_llm
from tfds_project.engine.reranker import get_reranker
from tfds_project.engine.graph import make_graph_agent

# Load environment variables in local mode
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception as e:
    pass


## BACK ##
# embeddings_function = get_embeddings_function()
embeddings_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)
llm = get_llm()
reranker = get_reranker("nano")
# vectorstore_graphs = Chroma(
#     persist_directory="C:/Users/dorab/OneDrive - Ecole Polytechnique/Documents/Data Science for Business HEC x Polytechnique/Year 2/Tooling for the Data Scientist/tfds_project/data/vectorstore",
#     embedding_function=embeddings_function,
# )
vectorstore_graphs = FAISS.load_local(
    "tests/faiss_index", embeddings_function, allow_dangerous_deserialization=True
)

agent = make_graph_agent(
    llm=llm,
    vectorstore_graphs=vectorstore_graphs,
    reranker=reranker,
)

## FRONT ##
st.title("Ask me anything about climate change !")

# Step 1: Create a text input field to capture user questions
user_input = st.text_input("Enter your question:")
inputs = {"user_input": user_input, "sources_input": "auto", "audience": "general"}

# Step 2: Add a button for submission
if st.button("Submit"):
    # Step 3: Simulate agent invocation (replace with actual call to agent)
    response = agent.invoke(inputs)

    st.write("Here are some interactive graphs related to your question:")

    # Extract the returned_content from recommended_content in the response
    recommended_content = response.get("recommended_content", [])

    # Step 4: Display the embedded returned_content (if any)
    if recommended_content:
        for doc in recommended_content:
            returned_content = doc.metadata.get("embedding", "")

            # Display each iframe content in the app (assuming returned_content is HTML)
            if returned_content:
                st.components.v1.html(returned_content, height=600, scrolling=True)
    else:
        st.write("No recommended content found.")


## TEST ##
# result = agent.invoke(
#     {
#         "user_input": "should I be a vegetarian ?",
#         "audience": "general",
#         "sources_input": "auto",
#     }
# )

# print(result)
