import os
import glob
from typing import List, Literal, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END

app = FastAPI(title="Zepto Support Assistant GenAI Service")

CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "zepto_policies"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"}
)

def ingest_documents():
    if collection.count() == 0:
        doc_files = glob.glob("docs/doc_*.txt")
        documents = []
        metadatas = []
        ids = []
        
        for file_path in doc_files:
            doc_id = os.path.basename(file_path).replace(".txt", "")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                documents.append(content)
                metadatas.append({"source": doc_id})
                ids.append(f"{doc_id}_chunk1")
                
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully ingested {len(documents)} documents into ChromaDB.")

ingest_documents()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

PROMPT_TEMPLATE = """
Role: You are an official Zepto Customer Support AI Assistant.
Context: {context}
Task: Answer the customer's question clearly and accurately based strictly on the provided context.
Format: Provide a direct, polite answer.
Length: Keep the response under 3 sentences.

Negative Constraint: Do not answer using information not present in the provided context. If the answer cannot be determined from the context, state that you do not have enough information.

Example 1:
Context: Zepto offers three account tiers: Basic, Zepto Pass (INR 49/month), and Zepto Pass+ (INR 99/month).
Question: How much is Zepto Pass?
Answer: Zepto Pass costs INR 49 per month.

Question: {query}
Answer:
"""

class AgentState(TypedDict):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: List[dict]
    response: QueryResponse

def classify_intent_node(state: AgentState) -> AgentState:
    query_lower = state["query"].lower()
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    
    if mock_mode:
        if any(kw in query_lower for kw in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        if any(kw in query_lower for kw in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"
            
    return {**state, "intent": intent}

def retrieve_and_answer_node(state: AgentState) -> AgentState:
    query = state["query"]
    mock_mode = os.getenv("MOCK_LLM", "1") == "1"
    
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    retrieved_ids = results["ids"][0] if results["ids"] else []
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
    if mock_mode:
        top_snippet = retrieved_docs[0][:200] if retrieved_docs else ""
        canned_answer = f"Based on the retrieved context: {top_snippet}"
        response = QueryResponse(
            answer=canned_answer,
            sources=retrieved_ids,
            confidence=1.0
        )
    else:
        top_snippet = retrieved_docs[0][:200] if retrieved_docs else ""
        response = QueryResponse(
            answer=f"Based on the retrieved context: {top_snippet}",
            sources=retrieved_ids,
            confidence=1.0
        )
        
    return {**state, "retrieved_chunks": retrieved_docs, "response": response}

def direct_answer_node(state: AgentState) -> AgentState:
    canned_answer = "I can only answer questions about Zepto policies right now."
    response = QueryResponse(
        answer=canned_answer,
        sources=[],
        confidence=1.0
    )
    return {**state, "response": response}

def route_intent(state: AgentState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
workflow.add_node("direct_answer", direct_answer_node)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": "general_question",
        "retrieved_chunks": [],
        "response": None
    }
    
    final_state = app_graph.invoke(initial_state)
    return final_state["response"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)