from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
COLLECTION = "hs_kb"

def build_vectorstore(historical_kb: list[dict]) -> Chroma:
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    for i, row in enumerate(historical_kb):
        text = f"{row['description']} , {row['category']}"
        docs.append(Document(
            page_content=text,
            metadata={"hs_code": row["hs_code"], "category": row["category"], "raw_description": row["description"], "doc_id": i,},
        ))

    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedder,
        persist_directory=PERSIST_DIR,
    )
    vs.add_documents(docs)
    vs.persist()
    return vs

def load_vectorstore() -> Chroma:
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embedder,
        persist_directory=PERSIST_DIR,
    )
