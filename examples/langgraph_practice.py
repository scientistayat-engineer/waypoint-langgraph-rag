"""Install: python -m pip install langgraph langchain-groq langchain-huggingface sentence-transformers
Run: python langgraph_practice.py
"""

import os
from getpass import getpass
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END

# 1. Set your Groq API key.
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")

# 2. Create sentence embeddings and store sample documents.
texts = [
    "LangGraph builds workflows using state, nodes, and edges.",
    "Sentence embeddings represent text as vectors that capture meaning.",
    "RAG retrieves relevant documents before generating an answer.",
]
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
store = InMemoryVectorStore.from_texts(texts, embedding=embeddings)
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# 3. State holds the data shared between nodes.
class State(TypedDict):
    question: str
    context: str
    answer: str

# 4. First node: retrieve relevant documents.
def retrieve(state: State):
    docs = store.similarity_search(state["question"], k=2)
    return {"context": "\n".join(doc.page_content for doc in docs)}

# 5. Second node: generate an answer.
def generate(state: State):
    reply = model.invoke([
        ("system", "Answer only using the supplied context. "
         "If it lacks the answer, say you don't know."),
        ("human", f"Context: {state['context']}\nQuestion: {state['question']}"),
    ])
    return {"answer": reply.content}

# 6. Connect the nodes and compile the graph.
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# 7. Run the graph.
question = input("Ask a question: ")
result = graph.invoke({"question": question, "context": "", "answer": ""})
print("\nAnswer:", result["answer"])
