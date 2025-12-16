import arxiv, os, time, hashlib, json
from datetime import datetime
from tqdm import tqdm

DATA_DIR = "/media/avaish/aiwork/local_llm/data/research"
CACHE_FILE = os.path.join(DATA_DIR, "arxiv_cache.json")

# 🔍 Define your research topics of interest
SEARCH_TOPICS = [
    "satellite image change detection",
    "visual search in satellite imagery",
    "semantic segmentation transformer",
    "DINOv2 SAM2 multimodal foundation models",
    "YOLO object detection evolution",
    "AI compiler optimization Triton kernel",
    "GPU performance profiling Nsight Compute",
    "NVIDIA GPU architecture Hopper Blackwell Tensor Core",
    "agentic GPU optimization AI workloads",
    "science AI for HPC and simulation",
    "TPU , JAX and Pallas Kernels",
    "AI Frameworks like Pytorch, Tensorflow",
    "AI Agentic systems like MCP , Langgraph, Langchain , LlamaIndex",
    "HSM , Cryptography and AI Security"
]


def md5(s: str):
    return hashlib.md5(s.encode()).hexdigest()


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def already_downloaded(paper_id: str, updated: str, cache: dict) -> bool:
    """Return True if this arXiv paper (same ID and version) already downloaded."""
    info = cache.get(paper_id)
    if not info:
        return False
    # arXiv paper URLs include version (v1, v2...), so detect change
    if info.get("updated") == updated:
        return True
    return False


def fetch_new_papers():
    """Fetch the latest relevant arXiv papers (avoiding duplicates)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = load_cache()
    new_papers = []

    for topic in SEARCH_TOPICS:
        print(f"\n🔍 Searching arXiv for: {topic}")
        search = arxiv.Search(
            query=topic,
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in search.results():
            paper_id = result.entry_id.split("/")[-1]  # e.g., "2407.12345v2"
            title = result.title.strip()
            updated = getattr(result, "updated", result.published).isoformat()

            if already_downloaded(paper_id, updated, cache):
                print(f"⏭️ Skipping (cached): {title[:60]}...")
                continue

            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
            pdf_filename = f"{paper_id}-{safe_title}.pdf"
            pdf_path = os.path.join(DATA_DIR, pdf_filename)

            try:
                print(f"  📥 Downloading: {title[:80]}...")
                result.download_pdf(filename=pdf_path)

                cache[paper_id] = {
                    "title": title,
                    "summary": result.summary[:400],
                    "authors": [a.name for a in result.authors],
                    "pdf": pdf_path,
                    "updated": updated,
                    "fetched": datetime.now().isoformat()
                }
                new_papers.append(cache[paper_id])
                save_cache(cache)
                time.sleep(1.5)
            except Exception as e:
                print(f"  ⚠️ Failed {title[:60]}: {e}")

    save_cache(cache)
    print(f"\n✅ {len(new_papers)} new papers added.")
    return new_papers


if __name__ == "__main__":
    new_papers = fetch_new_papers()
    if new_papers:
        print("\n🧩 Summary of new downloads:")
        for p in new_papers:
            print(f"- {p['title']} ({', '.join(p['authors'][:3])})")
            print(f"  ↳ {p['pdf']}\n")
    else:
        print("\nNo new papers found.")
