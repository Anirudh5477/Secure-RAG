# retrieval.py
import os
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Import our middleware layers
from Guardrails import scrub_pii, is_safe_intent
from cache_layer import cache
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv(), override=True)

print("🛡️ Initializing Final RAG Engine (with Source Tracking & Strict RBAC)...")

# 1. Connect to Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "enterprise_kb"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# 2. Use Groq Llama 3 70B Model for high quality reasoning
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

def secure_rag_query(user_query: str, user_role: str) -> str:
    """
    Executes the secure RAG sequence, strictly filtering by role,
    and returning the answer with file citations.
    """
    role = user_role.lower()
    
    # STAGE 1: THE BOUNCER (Intent Guardrail)
    if not is_safe_intent(user_query):
        return "❌ ACCESS DENIED: Request is off-topic or violates compliance."
        
    # STAGE 2: THE CENSOR (PII Eraser)
    clean_query = scrub_pii(user_query)
    
    # STAGE 3: CACHE CHECK (Strictly bound by department)
    cached_answer = cache.check_cache(clean_query, department=role)
    if cached_answer:
        return cached_answer
        
    # STAGE 4: ROLE-BASED DATABASE SEARCH
    # Qdrant will only search chunks where metadata.department == user_role
    rbac_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="metadata.department",
                match=qmodels.MatchValue(value=role)
            )
        ]
    )
    
    # Fetch top 3 relevant chunks
    matched_docs = vector_store.similarity_search(
        query=clean_query,
        k=6,
        filter=rbac_filter
    )
    
    # If 0 documents are found, they are either asking about data that doesn't exist
    # OR they are asking for another department's data. Deny them immediately.
    if not matched_docs:
        return f"❌ NOT AUTHORIZED: You do not have permission to view this data, or no such documents exist in the {role.upper()} repository."
        
    # STAGE 5: EXTRACT DATA & FILE SOURCES
    context_text = "\n\n".join([doc.page_content for doc in matched_docs])
    
    # Extract unique filenames using a Python set (so we don't print duplicates)
    file_sources = list(set([doc.metadata.get("file_name", "Unknown Document") for doc in matched_docs]))
    
    # STAGE 6: GOOGLE GEMMA / META LLM SYNTHESIS
    system_prompt = f"""You are a helpful, precise corporate AI. 
    Answer the user's query using ONLY the provided Authorized Context.
    Do not mention that you are reading from context. Just give the answer naturally.
    If the answer is not in the context, explicitly say you do not have that information.
    
    Authorized Context:
    {context_text}
    """
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clean_query}
    ])
    
    # --- NEW TELEMETRY & COST TRACKING ---
    # Extract exact token usage from Groq's metadata
    usage = response.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    # Synthetic Cost Calculation (Using typical Llama-3.3-70b rates)
    # ~$0.59 per 1M input tokens | ~$0.79 per 1M output tokens
    cost = (prompt_tokens * 0.59 / 1_000_000) + (completion_tokens * 0.79 / 1_000_000)
    
    print(f"\n📊 [TELEMETRY] Latency tracked in LangSmith")
    print(f"🪙 [TOKENS] Input: {prompt_tokens} | Output: {completion_tokens}")
    print(f"💰 [COST] Synthetic Request Cost: ${cost:.6f}\n")
    # -------------------------------------
    
    # Append the file sources to the bottom
    final_output = f"{response.content.strip()}\n\n📁 **Sourced from:** {', '.join(file_sources)}"
    
    # Save to Cache so the next person in this role gets it instantly
    cache.save_to_cache(clean_query, final_output, department=role)
    
    return final_output

# Visual Testing Block
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 TESTING SECURE RAG AUTHORIZATION")
    print("="*50)
    
    # Test Scenario: Both ask about Engineering Backend protocols
    test_query = "What is the primary tech stack used for the backend deployment?"
    
    print("\n[SCENARIO 1] 👨‍💻 Engineering User asks an Engineering Question")
    eng_response = secure_rag_query(user_query=test_query, user_role="engineering")
    print(f"🤖 Bot: {eng_response}")
    
    print("\n" + "-"*50)
    
    print("\n[SCENARIO 2] 📈 Marketing User asks an Engineering Question")
    mkt_response = secure_rag_query(user_query=test_query, user_role="marketing")
    print(f"🤖 Bot: {mkt_response}")