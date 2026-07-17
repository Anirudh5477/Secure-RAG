# security.py
import os
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider

# NEW IMPORTS FOR THE BOUNCER
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("⚙️ Initializing Security Middleware Engines...")

# 1. Define the custom configuration targeting the small model
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
}

# 2. Build the engine provider with the configuration
provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
nlp_engine = provider.create_engine()

# 3. Pass the customized engine into the Analyzer
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
anonymizer = AnonymizerEngine()

# Initialize the blazing-fast Groq model just for security checks
guardrail_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0, # Temperature 0 ensures strict, predictable classifications
    api_key=os.getenv("GROQ_API_KEY")
)

def scrub_pii(text: str) -> str:
    """The Censor: Scrubs sensitive PII from incoming queries."""
    if not text.strip(): return text
    
    analysis_results = analyzer.analyze(text=text, language="en")
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
        "US_SSN": OperatorConfig("replace", {"new_value": "<GOVT_ID>"}),
        "DATE_TIME": OperatorConfig("custom", {"lambda": lambda x: x}), # Let dates pass
    }
    
    result = anonymizer.anonymize(text=text, analyzer_results=analysis_results, operators=operators)
    return result.text

def is_safe_intent(query: str) -> bool:
    """
    The Bouncer: Evaluates if the query is relevant to internal corporate data.
    Returns True if safe, False if off-topic.
    """
    # We ask the LLM to reply with exactly one word to save time and tokens
    prompt = PromptTemplate.from_template(
        """You are a strict corporate security router.
        Evaluate the user's query and determine if it is relevant to a company workspace 
        (e.g., HR, Engineering, Finance, Marketing, company policies, general business).
        
        Rules:
        1. Queries about marketing, advertising, or business campaigns are SAFE (they are not political).
        2. A query that includes a greeting (e.g. "Hi") followed by a business question is SAFE.
        3. Simple standard greetings by themselves (e.g., "hi", "hello", "hey") are SAFE.
        4. If the query asks you to write code, asks for personal advice, discusses government politics, 
           or is purely random chit-chat with no business question, it is OFF-TOPIC (BLOCK).
        
        Reply with exactly one word: SAFE or BLOCK.
        
        User Query: {query}
        Decision:"""
    )
    
    chain = prompt | guardrail_llm | StrOutputParser()
    
    # Run the chain and strip whitespace/casing
    decision = chain.invoke({"query": query}).strip().upper()
    
    return "SAFE" in decision


# Visual Test Block
if __name__ == "__main__":
    print("\n--- Testing The Bouncer (Intent Guardrail) ---")
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Please add GROQ_API_KEY to your .env file to test the Bouncer!")
    else:
        test_queries = [
            "Where is the updated 2026 HR handbook?",
            "Can you write a Python script to scrape Twitter?",
            "What is our Q3 engineering budget?",
            "Who do you think will win the next political election?"
        ]
        
        for i, q in enumerate(test_queries, 1):
            is_safe = is_safe_intent(q)
            status = "✅ APPROVED" if is_safe else "❌ BLOCKED"
            print(f"\n[Test {i}] {status}")
            print(f"Query: {q}")
