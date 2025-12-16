import arxiv
import os
import time
import hashlib
import json
import random
from datetime import datetime

from domain_registry import DOMAINS

# =========================
# Configuration
# =========================

DATA_DIR = "/media/avaish/aiwork/local_llm/data/research"
CACHE_FILE = os.path.join(DATA_DIR, "arxiv_cache.json")

# Polite crawling parameters (arXiv-friendly)
ARXIV_REQUEST_DELAY = 4.0  # seconds between queries
ARXIV_BACKOFF_DELAY = 10.0  # seconds on failure
MAX_RESULTS_PER_QUERY = 10

# =========================
# Build search topics from domain registry
# =========================

SEARCH_TOPICS = sorted({
    topic
    for cfg in DOMAINS.values()
    for topic in cfg.get("arxiv_topics", [])
})


# =========================
# Helpers
# =========================

def normalize_arxiv_query(q: str) -> str:
    """
    Convert verbose natural-language queries into arXiv-friendly keyword queries.
    arXiv prefers short academic keyword expressions.
    """
    q = q.lower()

    # Remove filler words
    for junk in ["like", "systems", "and", ","]:
        q = q.replace(junk, "")

    # Canonical keyword whitelist
    keywords = [
        "cuda", "gpu", "triton", "nsight",
        "transformer", "vision", "segmentation",
        "satellite", "geospatial",
        "yolo", "foundation model",
        "agentic", "langchain", "langgraph",
        "cryptography", "hsm", "security",
        "jax", "pytorch", "tensorflow", "tunix"
    ]

    cleaned = [k for k in keywords if k in q]
    return " ".join(cleaned) if cleaned else q.strip()


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def already_downloaded(paper_id: str, updated: str, cache: dict) -> bool:
    """
    Return True if this arXiv paper (same ID and version timestamp) already exists.
    """
    info = cache.get(paper_id)
    return info is not None and info.get("updated") == updated


# =========================
# Main fetch logic
# =========================

def fetch_new_papers():
    """
    Fetch new arXiv papers across all configured domains.
    This function is resilient to arXiv outages and rate limits.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = load_cache()
    new_papers = []

    for topic in SEARCH_TOPICS:
        query = normalize_arxiv_query(topic)
        print(f"🔍 Searching arXiv for: {query}")

        try:
            search = arxiv.Search(
                query=query,
                max_results=MAX_RESULTS_PER_QUERY,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            for result in search.results():
                paper_id = result.entry_id.split("/")[-1]
                title = result.title.strip()
                updated = getattr(result, "updated", result.published).isoformat()

                if already_downloaded(paper_id, updated, cache):
                    continue

                safe_title = "".join(
                    c if c.isalnum() or c in " _-" else "_"
                    for c in title
                )[:80]

                pdf_filename = f"{paper_id}-{safe_title}.pdf"
                pdf_path = os.path.join(DATA_DIR, pdf_filename)

                try:
                    print(f"  📥 Downloading: {title[:80]}...")
                    result.download_pdf(filename=pdf_path)

                    cache[paper_id] = {
                        "title": title,
                        "summary": result.summary[:500],
                        "authors": [a.name for a in result.authors],
                        "pdf": pdf_path,
                        "updated": updated,
                        "fetched": datetime.now().isoformat(),
                    }

                    new_papers.append(cache[paper_id])
                    save_cache(cache)

                except Exception as e:
                    print(f"  ⚠️ PDF download failed: {title[:60]} | {e}")

            # Polite delay with jitter
            time.sleep(ARXIV_REQUEST_DELAY + random.uniform(0.5, 1.5))

        except Exception as e:
            # arXiv 429 / 503 / network failures
            print(f"⚠️ arXiv query failed for '{query}': {e}")
            print(f"⏳ Backing off for {ARXIV_BACKOFF_DELAY} seconds...")
            time.sleep(ARXIV_BACKOFF_DELAY)

    save_cache(cache)
    print(f"\n✅ {len(new_papers)} new papers added.")
    return new_papers


# =========================
# Standalone execution
# =========================

if __name__ == "__main__":
    papers = fetch_new_papers()
    if papers:
        print("\n🧩 Summary of new downloads:")
        for p in papers:
            print(f"- {p['title']} ({', '.join(p['authors'][:3])})")
            print(f"  ↳ {p['pdf']}\n")
    else:
        print("\nNo new papers found.")
