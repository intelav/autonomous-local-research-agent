import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# BASE = "/media/avaish/aiwork/local_llm"
# PAPER_DIR = os.path.join(BASE, "Papers")
# DB_DIR = os.path.join(BASE, "vector_db")

BASE_DIR = "/media/avaish/aiwork/local_llm"
DB_BASE = os.path.join(BASE_DIR, "vector_db")
DATA_DIR = os.path.join(BASE_DIR, "data")


def build_vector_db(domain):
    print(f"\n🔍 Indexing domain: {domain}")
    paper_dir = os.path.join(DATA_DIR, domain)
    db_dir = os.path.join(DB_BASE, domain)
    os.makedirs(db_dir, exist_ok=True)

    all_docs, total_pages, total_files = [], 0, 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    t0 = time.time()

    for root, _, files in os.walk(paper_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                try:
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    page_count = len(docs)
                    total_pages += page_count
                    total_files += 1
                    all_docs.extend(splitter.split_documents(docs))
                    print(f"  • {f:30} ({page_count:>3} pages)")
                except Exception as e:
                    print(f"  ⚠️ Skipping {f}: {e}")

    print(f"\n📄 Total PDFs: {total_files} | Total pages: {total_pages}")
    print(f"🕒 Time spent: {time.time() - t0:.2f}s")

    # Choose embedding per domain
    model = {
        "research": "intfloat/e5-base-v2",
        "education": "all-MiniLM-L6-v2",
        "recipes": "all-MiniLM-L6-v2"
    }.get(domain, "all-MiniLM-L6-v2")

    print(f"🔧 Using embedding model: {model}")

    embeddings = SentenceTransformerEmbeddings(model_name=model)
    vectordb = Chroma.from_documents(all_docs, embeddings, persist_directory=db_dir)
    vectordb.persist()

    print(f"✅ Vector DB built at: {db_dir}\n")
    return vectordb


def get_vector_db(domain):
    db_dir = os.path.join(DB_BASE, domain)
    if os.path.exists(db_dir):
        model = "intfloat/e5-base-v2" if domain == "research" else "all-MiniLM-L6-v2"
        embeddings = SentenceTransformerEmbeddings(model_name=model)
        print(f"🔁 Loading existing DB from: {db_dir}")
        return Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        return build_vector_db(domain)


def get_qa_chain(domain="research", model_name="llama3:8b-instruct-q4_K_M"):
    vectordb = get_vector_db(domain)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8, "score_threshold": 0.35})
    llm = Ollama(model=model_name, base_url="http://localhost:11434")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa
