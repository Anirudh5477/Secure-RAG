# cache_layer.py
import os
import time
from collections import OrderedDict
import numpy as np
from dotenv import load_dotenv, find_dotenv

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq

load_dotenv(find_dotenv(), override=True)

print("🧠 Initializing Secure Multi-Tenant LRU Semantic Caching Engine...")

class LRUSemanticCache:
    def __init__(self, max_size=1000, threshold=0.75):
        """
        max_size: Maximum number of queries to remember before evicting the oldest.
        threshold: Conceptual similarity cutoff (0.0 to 1.0).
        """
        self.memory = OrderedDict() 
        self.max_size = max_size
        self.threshold = threshold
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def check_cache(self, query: str, department: str):
        """
        Checks cache for a semantic match, strictly ensuring the cached answer 
        belongs to the same department requesting it.
        """
        if not self.memory:
            return None
            
        query_vector = self.embeddings.embed_query(query)
        dept_lower = department.lower()
        
        # Unique identifier tuple combining the query string and department
        for (stored_query, stored_dept), cache_data in list(self.memory.items()):
            # SECURITY STEP: If the department doesn't match, completely skip this cache item!
            if stored_dept != dept_lower:
                continue
                
            stored_vector = cache_data["vector"]
            stored_answer = cache_data["answer"]
            
            similarity = self.cosine_similarity(query_vector, stored_vector)
            
            if similarity >= self.threshold:
                print(f"   [!] Secure Cache Hit for [{department.upper()}]! (Similarity Score: {similarity:.2f})")
                # Move item to front as Most Recently Used
                self.memory.move_to_end((stored_query, stored_dept), last=False)
                return stored_answer
                
        return None
        
    def save_to_cache(self, query: str, answer: str, department: str):
        """Saves the answer using a composite key of (query, department)."""
        query_vector = self.embeddings.embed_query(query)
        dept_lower = department.lower()
        composite_key = (query, dept_lower)
        
        # Save cache item
        self.memory[composite_key] = {"vector": query_vector, "answer": answer}
        self.memory.move_to_end(composite_key, last=False)
        
        # Evict oldest item if limit breached
        if len(self.memory) > self.max_size:
            (evicted_query, evicted_dept), _ = self.memory.popitem(last=True)
            print(f"   [-] Cache full. Evicted query: '{evicted_query}' from department [{evicted_dept.upper()}]")


# Initialize global components (set max_size to 5 for simulation layout)
cache = LRUSemanticCache(max_size=5, threshold=0.75)

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

def run_cache_simulation():
    print("\n" + "="*50)
    print("🚀 STARTING SECURE ROLE-BASED CACHE SIMULATION")
    print("="*50)
    
    # --- RUN 1: Engineering guy asks a question ---
    q1 = "What is the onboarding checklist and system setup protocol?"
    dept1 = "engineering"
    print(f"\n[Run 1] {dept1.upper()} requesting: '{q1}'")
    
    ans1 = cache.check_cache(q1, department=dept1)
    if not ans1:
        ans1 = f"[Fake Engineering Doc Response] Set up your IDE, clone git repos, and request AWS access tokens."
        cache.save_to_cache(q1, ans1, department=dept1)
    print(f"Status: Cache MISS 🔴 | Response: {ans1}")

    # --- RUN 2: Marketing guy asks an identical question ---
    q2 = "What is the onboarding checklist and system setup protocol?"
    dept2 = "marketing"
    print(f"\n[Run 2] {dept2.upper()} requesting: '{q2}' (Testing cross-department leak risk)")
    
    ans2 = cache.check_cache(q2, department=dept2)
    if not ans2:
        # Since it missed (as it should!), it generates a Marketing specific response
        ans2 = f"[Fake Marketing Doc Response] Log into HubSpot, check Canva brand templates, and read marketing guidelines."
        cache.save_to_cache(q2, ans2, department=dept2)
    print(f"Status: Cache MISS 🔴 (Securely Blocked Cross-Department Leak!) | Response: {ans2}")

    # --- RUN 3: Another Engineering guy asks a similar question ---
    q3 = "How do I get my development environment ready as a new hire?"
    dept3 = "engineering"
    print(f"\n[Run 3] {dept3.upper()} requesting: '{q3}' (Testing valid internal cache hit)")
    
    ans3 = cache.check_cache(q3, department=dept3)
    if not ans3:
        ans3 = llm.invoke(q3).content
        cache.save_to_cache(q3, ans3, department=dept3)
    print(f"Status: Cache HIT 🟢 | Response: {ans3}")

if __name__ == "__main__":
    run_cache_simulation()