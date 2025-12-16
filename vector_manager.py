import os, json, hashlib, time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

BASE_DIR = "/media/avaish/aiwork/local_llm"
DB_BASE = os.path.join(BASE_DIR, "vector_db")
DATA_BASE = os.path.join(BASE_DIR, "data")


def compute_file_hash(path, chunk_size=8192):
    """Compute stable MD5 hash for a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def needs_update(path, cache, tolerance=1.0):
    """Return True if file content changed or not in cache."""
    if not os.path.exists(path):
        return False
    hash_val = compute_file_hash(path)
    info = cache.get(path)
    if info is None:
        return True
    # Check content change
    if info.get("hash") != hash_val:
        return True
    # Ignore minor timestamp drift
    old_time = info.get("mtime", 0)
    new_time = os.path.getmtime(path)
    if abs(new_time - old_time) > tolerance:
        return True
    return False

def sanitize_documents(docs):
    clean = []
    for d in docs:
        text = d.page_content
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < 20:
            continue
        d.page_content = text
        clean.append(d)
    return clean

def build_or_update_vector_db(domain):
    """Incrementally update or reuse an existing domain vector DB."""
    print(f"\n🔍 Checking domain: {domain}")
    paper_dir = os.path.join(DATA_BASE, domain)
    db_dir = os.path.join(DB_BASE, domain)
    os.makedirs(db_dir, exist_ok=True)

    # Each domain keeps its own hash cache file
    cache_file = os.path.join(db_dir, "file_hash_cache.json")
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}

    updated_files = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    total_new_chunks = 0

    # Detect new/changed PDFs
    for root, _, files in os.walk(paper_dir):
        for f in files:
            if not f.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, f)
            if needs_update(path, cache):
                print(f"  ⚙️ Updating embeddings for {f}")
                try:
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    # chunks = splitter.split_documents(docs)
                    # total_new_chunks += len(chunks)
                    # updated_files.append((path, chunks))
                    raw_chunks = splitter.split_documents(docs)
                    chunks = sanitize_documents(raw_chunks)
                    if not chunks:
                        print(f"  ⚠️ No valid text chunks in {f}")
                        continue

                    total_new_chunks += len(chunks)
                    updated_files.append((path, chunks))
                    cache[path] = {
                        "mtime": os.path.getmtime(path),
                        "hash": compute_file_hash(path),
                        "updated": time.ctime(),
                    }
                except Exception as e:
                    print(f"  ⚠️ Skipping {f}: {e}")

    embeddings = SentenceTransformerEmbeddings(model_name="intfloat/e5-base-v2")
    vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    if not updated_files:
        print("✅ No new or changed PDFs detected — using cached embeddings.")
        return vectordb

    print(f"🔧 {len(updated_files)} files changed — re-embedding {total_new_chunks} chunks...")

    for path, chunks in updated_files:
        good, bad = 0, 0
        for c in chunks:
            try:
                vectordb.add_documents([c])
                good += 1
            except Exception as e:
                bad += 1
        if bad:
            print(f"⚠️ {os.path.basename(path)}: {bad} chunks skipped, {good} embedded")
    vectordb.persist()

    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"✅ Updated DB: {db_dir} | {len(updated_files)} PDFs re-indexed.")
    return vectordb
