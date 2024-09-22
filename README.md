# TFDS Climate Question App

## Overview

The **TFDS Climate Question App** is an interactive web-based application that allows users to ask questions about **climate change** and subjects related to the environment, nature, etc. and receive relevant **graphs** in response. The questions can be asked in any language. These graphs are sourced from a trusted climate data provider **Our World in Data** , helping users visualize the impact of climate change on various factors such as energy consumption, greenhouse gas emissions, and more. It leverages external data sources like **Our World in Data** to provide interactive and insightful visualizations.

This app is highly relevant because it improves accessibility to the graphs provided by **Our World in Data**, increasing climate literacy thereby raising awareness about the negative impacts of climate change.

The graphs were scraped from the https://ourworldindata.org/data in a separate repository.

The app uses **OpenAI embeddings**, **a custom retriever**, and a **reranking system** to fetch the most relevant graphs based on the user's input.

## Features

- **Natural Language Question Input**: Users can ask questions in plain English about climate change or related topics.
- **Relevant Graphs and Visualizations**: The app returns relevant graphs based on the question, allowing users to visualize climate data (e.g., carbon emissions, energy consumption, global temperature trends).
- **Efficient Retrieval**: The app uses embeddings and vector similarity search to retrieve the most relevant content.
- **Data Sources**: Trusted sources like **Our World in Data** are used for all visualizations.

## How It Works

1. **User Input**: A user inputs a question (e.g., "What are the effects of being vegetarian on the environment?").
2. **Agent Processing**: The app invokes a specialized agent that processes the user's question, identifies the key topics, and retrieves relevant content. During the process it does query translation, transformation and expansion to improve the results.
3. **Reranking and Retrieval**: The content is ranked and filtered based on relevance, and graphs that best match the user's question are selected.
4. **Display**: The app embeds and displays these graphs, offering interactive exploration.
