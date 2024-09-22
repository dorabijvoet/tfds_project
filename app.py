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
embeddings_function = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)
llm = get_llm()
reranker = get_reranker("nano")
vectorstore_graphs = FAISS.load_local(
    "tests/faiss_index", embeddings_function, allow_dangerous_deserialization=True
)

agent = make_graph_agent(
    llm=llm,
    vectorstore_graphs=vectorstore_graphs,
    reranker=reranker,
)

## FRONT ##
# Custom CSS for full-width layout and button styling
st.markdown(
    """
    <style>
    /* Make the Streamlit app use the full width of the page */
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
        max-width: 85%;  /* Set max width to 100% */
    }

    /* Style the button */
    div.stButton > button {
        display: block;
        margin: auto;  /* Center the button */
        background-color: #4CAF50; /* Green background */
        color: white;
        padding: 15px 32px;
        font-size: 20px;
        border-radius: 12px;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #45a049;  /* Darker green when hovered */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ask me anything about climate change in any language!")

# Step 1: Create a form that allows submission via both a button and pressing Enter
with st.form(key="question_form"):
    user_input = st.text_input("Enter your question:")
    inputs = {"user_input": user_input, "sources_input": "auto", "audience": "general"}

    # Step 2: Add the submit button within the form
    submit_button = st.form_submit_button("Submit")


# Step 2: Add a button for submission
if submit_button:
    # Step 3: Simulate agent invocation (replace with actual call to agent)
    response = agent.invoke(inputs)

    st.markdown(
        '<p style="color:darkblue; font-weight:bold;">Here are some interactive graphs related to your question:</p>',
        unsafe_allow_html=True,
    )

    # Extract the returned_content from recommended_content in the response
    recommended_content = response.get("recommended_content", [])

    # Step 4: Display the embedded returned_content (if any)
    if recommended_content:
        # Create a dictionary to group iframes by category
        categorized_content = {}
        for doc in recommended_content:
            category = doc.metadata.get("category", "Other")
            iframe_content = doc.metadata.get("embedding", "")

            # Add iframe content to the respective category
            if category in categorized_content:
                categorized_content[category].append(iframe_content)
            else:
                categorized_content[category] = [iframe_content]

        # Step 6: Create tabs for each category
        if categorized_content:
            tabs = st.tabs(
                list(categorized_content.keys())
            )  # Create a tab for each category

            # Loop through each category and corresponding tab
            for i, (category, iframes) in enumerate(categorized_content.items()):
                with tabs[i]:  # Display content in the respective tab

                    # Dynamically create a layout based on the number of iframes
                    num_iframes = len(iframes)

                    if num_iframes == 1:
                        # If only one iframe, display it full width
                        st.components.v1.html(iframes[0], height=600, scrolling=True)
                    elif num_iframes == 2:
                        # If two iframes, display them side by side (2 columns)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.components.v1.html(
                                iframes[0], height=600, scrolling=True
                            )
                        with col2:
                            st.components.v1.html(
                                iframes[1], height=600, scrolling=True
                            )
                    else:
                        # For three or more iframes, display in a grid with up to 3 columns per row
                        num_cols = 3  # Set number of columns per row
                        cols = st.columns(num_cols)

                        for idx, iframe in enumerate(iframes):
                            with cols[idx % num_cols]:  # Cycle through columns
                                st.components.v1.html(
                                    iframe, height=600, scrolling=True
                                )
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
