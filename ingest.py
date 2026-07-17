import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

print("⏳ [Step 0] Script started! Loading dependencies and environment variables...")

# Added CSVLoader to handle spreadsheet data cleanly
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "enterprise_kb"

def generate_deterministic_id(doc_content: str, file_name: str) -> str:
    unique_string = f"{file_name}_{doc_content}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

def main():
    data_dir = Path("data")
    print(f"\n📁 [Step 1] Checking data directory: {data_dir.resolve()}")
    
    if not data_dir.exists():
        print("❌ ERROR: The 'data' folder does not exist in this directory!")
        return

    raw_docs = []
    valid_extensions = [".txt", ".md", ".markdown", ".csv"]
    
    for folder in data_dir.iterdir():
        if folder.is_dir():
            department = folder.name.lower()
            print(f"   └─ Scanning folder: [{department}]")
            
            for file_path in folder.rglob("*"):
                ext = file_path.suffix.lower()
                if ext in valid_extensions:
                    print(f"      ├─ Found file: {file_path.name}")
                    try:
                        # Route to the appropriate loader based on file extension
                        if ext == ".csv":
                            loader = CSVLoader(str(file_path), encoding="utf-8")
                        else:
                            loader = TextLoader(str(file_path), encoding="utf-8")
                            
                        docs = loader.load()
                        
                        # Attach RBAC security badges to all loaded items/rows
                        for doc in docs:
                            doc.metadata["department"] = department
                            doc.metadata["file_name"] = file_path.name
                        raw_docs.extend(docs)
                    except Exception as e:
                        print(f"      ❌ Error reading {file_path.name}: {e}")

    print(f"\n📊 Total documents/rows loaded: {len(raw_docs)}")
    
    if len(raw_docs) == 0:
        print("⚠️ WARNING: No valid files found!")
        return

    print("\n✂️ [Step 2] Slicing documents into chunks with Markdown & Recursive splitters...")
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    chunked_docs = []
    for doc in raw_docs:
        # 1. Markdown Split (returns Langchain Documents with header metadata)
        md_splits = markdown_splitter.split_text(doc.page_content)
        
        # 2. CRITICAL: Re-attach RBAC metadata (department, file_name) and any other existing metadata
        for split in md_splits:
            split.metadata.update(doc.metadata)
            
        # 3. Secondary Recursive Split
        final_splits = text_splitter.split_documents(md_splits)
        chunked_docs.extend(final_splits)

    print(f"Created {len(chunked_docs)} total chunks.")

    print("\n🧠 [Step 3] Initializing local BGE embedding model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("\n☁️ [Step 4] Connecting to Qdrant Cloud...")
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ ERROR: QDRANT_URL or QDRANT_API_KEY is missing from your .env file!")
        return
        
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if client.collection_exists(COLLECTION_NAME):
        print(f"🗑️ Dropping existing collection '{COLLECTION_NAME}' to prevent duplicate/stale chunks...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating new Qdrant collection: '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    print("Generating deterministic UUIDs to prevent duplicates...")
    chunk_ids = [
        generate_deterministic_id(doc.page_content, doc.metadata["file_name"]) 
        for doc in chunked_docs
    ]

    print(f"🚀 Uploading {len(chunked_docs)} chunks to Qdrant Cloud...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_documents(documents=chunked_docs, ids=chunk_ids)

    print("\n✅ SUCCESS! All markdown and CSV files ingested and secured with RBAC metadata!")

if __name__ == "__main__":
    main()