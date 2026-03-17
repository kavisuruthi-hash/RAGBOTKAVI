"""
rag_agent.py - LangGraph RAG Agent with Citations

This module implements a simple RAG (Retrieval-Augmented Generation) agent
using LangGraph. The agent follows this flow:

    [User Query] -> [Retrieve Chunks] -> [Generate Answer + Citations] -> [Final Response]

The graph has two nodes:
    1. retrieve  - Queries the vector DB for relevant document chunks
    2. generate  - Sends the chunks + question to the LLM for a grounded answer

WHAT TO MODIFY (for students):
    - SYSTEM_PROMPT  : Change the agent's personality and instructions
    - TOP_K          : How many chunks to retrieve (more = broader context)
    - LLM_MODEL      : Change the OpenAI model used for generation
    - TEMPERATURE    : 0 = deterministic, 1 = creative
    - retrieve()     : Try different retrieval methods (see comments inside)
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from ingestion import CHROMA_DB_DIR, EMBEDDING_MODEL


# ==========================================================================
# CONFIGURATION - MODIFY HERE
# ==========================================================================

# How many chunks to retrieve per query
# - More chunks (5-8) = broader context, may include less relevant info
# - Fewer chunks (2-3) = more focused, but may miss relevant info
TOP_K = 3

# --------------------------------------------------------------------------
# LLM SETTINGS - Students: experiment with these!
# --------------------------------------------------------------------------
# Model to use for generating answers. Options:
#   "gpt-4.1-mini"     - Fast, cheap, good quality (default)
#   "gpt-4.1-nano"     - Fastest, cheapest, lower quality
#   "gpt-4o-mini"      - Previous gen, fast and cheap
#   "gpt-4o"           - High quality, slower, more expensive
LLM_MODEL = "gpt-4.1-mini"

# Temperature controls randomness:
#   0.0 = deterministic (same answer every time) - best for factual Q&A
#   0.7 = creative (varied answers) - better for brainstorming
#   1.0 = very creative (may hallucinate more)
TEMPERATURE = 0

# --------------------------------------------------------------------------
# SYSTEM PROMPT - Students: this is the most fun part to modify!
# --------------------------------------------------------------------------
# This prompt tells the LLM HOW to behave and HOW to use the retrieved context.
# Try changing the tone, adding rules, or making it domain-specific.
#
# EXAMPLES TO TRY:
#   - "You are a medical expert. Use clinical terminology..."
#   - "You are a tutor. Explain concepts simply for a 10-year-old..."
#   - "You are a legal assistant. Always cite specific sections..."
#   - "Answer in bullet points only."
#   - "If you're not sure, list what you DO know and what's missing."
#
SYSTEM_PROMPT = """You are a friendly and helpful customer support agent. Answer the customer's \
question based ONLY on the following context from retrieved documents.

RULES:
## Response Format

Always structure your responses as follows:

1. ACKNOWLEDGE the customer's issue with empathy in 1–2 sentences.
2. STEPS — provide a numbered, step-by-step solution using ONLY information from the context below.
3. CONFIRMATION CHECK — end with a short "Does that help?" or similar.
4. "Need more help?" — list 2–3 suggested follow-up questions.
5. SOURCES — list which documents and pages you used.

---

## Writing Style

- Use plain, everyday language — avoid technical jargon.
- If a technical term is unavoidable, explain it in parentheses.
  Example: "Clear your cache (the temporary files your browser stores)"
- Keep sentences short. One idea per sentence.
- Use "you" and "we" to keep it conversational.
- Never use bullet points for the main steps — always use numbered lists.

---

## Step-by-Step Instructions Format

Always number steps clearly starting from 1. Example:
1. Open the app on your phone.
2. Tap the menu icon (the three lines in the top-left corner).
3. Select Settings, then tap Account.
4. Tap Save when you're done.

---

## "Need more help?" Section (Required)

At the end of every response, add this section in exactly this format:

Need more help?
You might also want to know:
- [Suggested question 1 related to the topic]
- [Suggested question 2 related to the topic]
- [Suggested question 3 related to the topic]

Generate these questions based on what a customer might naturally ask next.

---

## Sources Section (Required)

After the "Need more help?" section, always add a Sources section in exactly this format:

Sources:
- Document: [document name], Page: [page number]
- Document: [document name], Page: [page number]

Only list documents and pages that were actually used in your answer.

---

## Tone & Empathy Rules

- If the customer sounds frustrated, acknowledge it first.
  Example: "I completely understand how frustrating this can be — let's get it sorted right away."
- Never blame the customer.
- If the context lacks information, say: "I don't have that information in my current \
sources, but here's how you can get it quickly: …"

---

## RULES

1. Only use information from the provided context below.
2. If the context does not contain enough information, say so honestly.
3. Never make up product features, policies, or source references.
4. Do not ask more than one clarifying question at a time.

---

Context:
{context}

Source Documents:
{sources}"""


# ==========================================================================
# STATE DEFINITION
# ==========================================================================
# LangGraph agents pass a shared "state" dict between nodes.
# We define the shape of that state here using TypedDict.

class RAGState(TypedDict):
    question: str   # The user's question
    context: str    # Retrieved document chunks (joined text)
    sources: str    # Citation info (file names + page numbers)
    answer: str     # The final generated answer


# ==========================================================================
# NODE 1: RETRIEVE - Query the vector DB for relevant chunks
# ==========================================================================

def retrieve(state: RAGState) -> dict:
    """
    RAG Step 1 - Query the vector DB.

    Takes the user's question, performs a similarity search against
    ChromaDB, and returns the top-K most relevant chunks as context,
    along with source citations (file name + page number).
    """
    question = state["question"]

    # Load the persisted ChromaDB vector store (same local model as ingestion)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )

    # --------------------------------------------------------------------------
    # MODIFY HERE: Try different retrieval methods
    # --------------------------------------------------------------------------

    # METHOD 1: Basic similarity search (DEFAULT)
    # Returns the TOP_K chunks most similar to the question
    results = vector_store.similarity_search(question, k=TOP_K)

    # METHOD 2: Similarity search with relevance scores
    # Returns chunks along with their similarity scores (0 to 1)
    # Uncomment below and comment out Method 1 to try it:
    #
    # results_with_scores = vector_store.similarity_search_with_relevance_scores(
    #     question, k=TOP_K
    # )
    # # Filter out low-relevance chunks (score < 0.3)
    # results = [doc for doc, score in results_with_scores if score > 0.3]
    # for doc, score in results_with_scores:
    #     print(f"    Score: {score:.3f} | {doc.metadata.get('source', '?')}")

    # METHOD 3: MMR (Maximum Marginal Relevance)
    # Balances relevance with diversity - avoids returning similar chunks
    # Uncomment below to try it:
    #
    # results = vector_store.max_marginal_relevance_search(
    #     question, k=TOP_K, fetch_k=TOP_K * 3
    # )

    # --------------------------------------------------------------------------

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Build citation info from chunk metadata
    # Each chunk carries metadata from ingestion: {"source": "data/file.pdf", "page": 0}
    sources_list = []
    for i, doc in enumerate(results, 1):
        source_file = doc.metadata.get("source", "Unknown")
        page_num = doc.metadata.get("page", "?")
        sources_list.append(f"  [{i}] {source_file} (Page {page_num + 1 if isinstance(page_num, int) else page_num})")

    sources = "\n".join(sources_list)

    print(f"  [Retrieve] Found {len(results)} relevant chunks for: '{question}'")
    for s in sources_list:
        print(f"  {s}")

    return {"context": context, "sources": sources}


# ==========================================================================
# NODE 2: GENERATE - Answer the question using LLM + retrieved context
# ==========================================================================

def generate(state: RAGState) -> dict:
    """
    RAG Step 2 - Answer from RAG chunks with citations.

    Takes the retrieved context, source citations, and the original question.
    Sends them to the LLM with the system prompt to produce a grounded answer.
    """
    question = state["question"]
    context = state["context"]
    sources = state["sources"]

    # Build the prompt from our configurable SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Initialize the LLM with configured model and temperature
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

    # Build the chain: prompt -> LLM
    chain = prompt | llm

    # Invoke the chain with our context, sources, and question
    response = chain.invoke({
        "context": context,
        "sources": sources,
        "question": question,
    })

    print(f"  [Generate] Answer produced.")
    return {"answer": response.content}


# ==========================================================================
# BUILD THE LANGGRAPH AGENT
# ==========================================================================

def build_rag_agent() -> StateGraph:
    """
    Constructs the LangGraph RAG agent with two nodes:

        START -> retrieve -> generate -> END

    This is intentionally kept simple to clearly show the RAG flow:
    1. User asks a question
    2. retrieve node fetches relevant chunks from ChromaDB
    3. generate node uses those chunks + citations to produce a grounded answer
    4. The answer (with source references) is returned
    """
    # Create the state graph with our RAGState schema
    graph = StateGraph(RAGState)

    # Add the two nodes
    graph.add_node("retrieve", retrieve)   # Node 1: Query vector DB
    graph.add_node("generate", generate)   # Node 2: Generate answer

    # Define the edges (flow): START -> retrieve -> generate -> END
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Compile the graph into a runnable agent
    agent = graph.compile()
    return agent


def query_rag(question: str) -> str:
    """
    Run a single question through the RAG agent.

    Args:
        question: The user's question string.

    Returns:
        The agent's answer (with citations) as a string.
    """
    agent = build_rag_agent()

    # Run the agent with the initial state
    result = agent.invoke({
        "question": question,
        "context": "",     # Will be filled by the retrieve node
        "sources": "",     # Will be filled by the retrieve node
        "answer": "",      # Will be filled by the generate node
    })

    return result["answer"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Quick test
    answer = query_rag("What is this document about?")
    print(f"\nAnswer: {answer}")
