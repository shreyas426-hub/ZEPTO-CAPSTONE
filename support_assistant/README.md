\# Module 3: Zepto Support Assistant (GenAI RAG Pipeline)



\## Overview

A complete RAG-based customer support assistant for Zepto built using FastAPI, LangGraph, ChromaDB, and Sentence-Transformers.



\## Pipeline Architecture Description

1\. \*\*Ingestion:\*\* Text documents from `docs/` (`doc\_01.txt` to `doc\_08.txt`) are ingested and stored as chunks.

2\. \*\*Embedding:\*\* Embeddings are generated locally with `all-MiniLM-L6-v2` via `sentence-transformers`.

3\. \*\*Retrieval:\*\* Vector search runs in local ChromaDB (`zepto\_policies`) returning top matching policy chunks.

4\. \*\*Generation:\*\* LangGraph `StateGraph` routes queries via `classify\_intent` to `retrieve\_and\_answer` or `direct\_answer`.



\### MOCK\_LLM Behavior

\- \*\*`MOCK\_LLM=1` (Default Graded Baseline):\*\* Runs offline with rule-based heuristics and structured output without making network/API calls.

\- \*\*`MOCK\_LLM=0` (Optional Real LLM):\*\* Connects to external LLM provider.



\---



\## Local Run \& Test Examples (Default Mode)



\### Start Server

```cmd

python main.py

