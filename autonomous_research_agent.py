import os, time, textwrap, json
from datetime import datetime
import random

from agent_local_rag import get_vector_db
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from arxiv_fetcher import fetch_new_papers
from vector_manager import build_or_update_vector_db
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tabulate import tabulate
import re

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
# === Config ===
DOMAIN = "research"
REFRESH_INTERVAL = 30  # min
WRAP_WIDTH = 100


def format_answer(ans: str) -> str:
    """Format LLM answers into clean, bullet-based Markdown (no tables)."""
    if not ans:
        return "_No answer provided._"

    ans = ans.strip()

    # Normalize common list indicators
    ans = re.sub(r"(?m)^\s*\d+\.\s+", "- ", ans)  # numbered lists -> bullets
    ans = re.sub(r"(?m)^\s*[•▪]\s+", "- ", ans)  # bullet dots -> '- '

    # Highlight domain-specific keywords
    ans = re.sub(
        r"\b(Triton|CUDA|GPU|Nsight|kernel|optimization|transformer|inference|compiler|agentic|segmentation|foundation|DINOv2|SAM2|Convolution)\b",
        r"**\1**",
        ans,
        flags=re.IGNORECASE,
    )

    # Split into paragraphs or bullet blocks
    lines = [line.strip() for line in ans.split("\n") if line.strip()]
    formatted = []

    for line in lines:
        # Ensure bullets are clearly separated
        if line.startswith("- "):
            formatted.append(line)
        else:
            formatted.append(textwrap.fill(line, width=100))

    # Add spacing between bullet groups and paragraphs
    formatted_text = "\n\n".join(formatted)

    # Optional: emphasize section-like lines (like “Data Preparation: …”)
    formatted_text = re.sub(
        r"(?m)^- ([A-Z][A-Za-z ]+):",
        lambda m: f"- **{m.group(1)}:**",
        formatted_text
    )

    return formatted_text


def classify_domain(query: str):
    """Assign a domain tag based on keywords in the query."""
    tags = []
    q = query.lower()
    if any(k in q for k in ["cuda", "gpu", "triton", "kernel", "nsight"]):
        tags.append("gpu_optimization")
    if any(k in q for k in ["segmentation", "mask", "sam", "dinov2", "vit", "transformer", "convolution"]):
        tags.append("foundation_models")
    if any(k in q for k in ["satellite", "geospatial", "remote", "visual search", "change detection"]):
        tags.append("geospatial_ai")
    if any(k in q for k in ["agentic", "autonomous", "optimization", "profiling"]):
        tags.append("agentic_gpu")
    if not tags:
        tags.append("general")
    return tags


def enhance_paragraphs(chunks, llm):
    """Fuse and smooth partial paragraphs for readability."""
    text = "\n".join(chunks)
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Rewrite the following extracted technical paragraphs so that they form a coherent, "
            "complete explanation with no broken sentences. Keep it factual and concise:\n\n{text}"
        ),
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(text)


def show_top_hits(query, vectordb, run_dir, llm=None):
    docs_with_scores = vectordb.similarity_search_with_score(query, k=5)
    rows = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        snippet = doc.page_content.strip().replace("\n", " ")
        rows.append((i, score, src, snippet[:800]))
    print("\n🔎 Top retrieved chunks:")
    print(tabulate(rows, headers=["#", "Score", "Source", "Snippet"], tablefmt="fancy_grid",
                   maxcolwidths=[None, None, None, 100]))

    context_dir = os.path.join(run_dir, "contexts")
    os.makedirs(context_dir, exist_ok=True)
    tag = classify_domain(query)
    #safe_tag = tag.replace(" ", "_")

    ctx_file = os.path.join(context_dir, f"context_{tag}_{datetime.now().strftime('%H%M%S')}.md")
    with open(ctx_file, "w") as cf:
        cf.write("# Retrieved Contexts\n\n")
        for i, (_, score, src, snippet) in enumerate(rows, 1):
            cf.write(f"## {i}. {src} (score {score:.3f})\n\n{snippet}\n\n---\n")
    if llm:
        enriched = enhance_paragraphs([r[-1] for r in rows], llm)
        print("\n📘 Enhanced Context Summary:\n")
        print(textwrap.fill(enriched, width=100))


domain_expertise = [
    "CUDA Programming and Custom kernels including Triton Kernels",
    "Object Detection and Segmentation, Vision Transformer",
    "AI frameworks like Pytorch, JAX, Tensorflow and their important features and latest advancements",
    "Foundation models(Transformer, DINOv2, SAM2, Whisper, LLAMA)",
    "GPU compiler and profiling techniques",
    "Data science using numpy and panda",
    "Latest in YOLO, Its performance and Loss Function",
    "machine learning to use various algorithm like SVM, naive bayes classifier, random forest, XGBoost",
    "Deep learning, CNN , RNN, LSTM, Reinforcement learning , GAN",
    "AI Agentic systems like MCP , Langgraph, Langchain , LlamaIndex",
    "HSM , Cryptography and AI Security"
]


def load_dynamic_queries():
    """Combine user-collected queries + core research prompts."""
    # base_queries = [
    #     "What new methods are proposed for semantic segmentation?",
    #     "Summarize key differences between YOLOv1–v10 evolution.",
    #     "Explain how DINOv2 and SAM2 complement each other.",
    #     "What are the latest GPU optimization insights from NVIDIA guides?",
    #     "List key advances in transformer-based vision models.",
    #     "Fundamentals of Convolution and Segmentation Network "
    #
    # ]
    base_queries = domain_expertise
    log_file = "user_queries.log"
    user_queries = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
        # Take most recent 20 distinct user queries
        user_queries = sorted(set([l.split("|", 1)[-1].strip() for l in lines]))[-20:]
    # Add your active research interests dynamically
    ai_focus = [
        "Recent papers on visual search in satellite imagery",
        "New approaches for change detection using foundation models",
        "Efficient CUDA kernel profiling methods using Nsight Compute",
        "New Triton compiler optimization techniques for AI inference",
        "Advances in agentic GPU systems or autonomous GPU tuning",
        "New multimodal or geospatial foundation models like SAM2, DINOv2, and Contrastive Models",
        "AI with Security"
    ]
    return base_queries + user_queries + ai_focus


def generate_domain_queries(seed_queries, query_dir, n=6):
    """Use LLM to generate new research queries dynamically."""
    try:
        llm = Ollama(model="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434")
    except Exception as e:
        print(f"⚠️ LLM initialization failed: {e}")
        return

    prompt = f"""
    You are an autonomous research assistant .
    Your mission: propose new research queries that will help deepen knowledge and make me expert in:
    {domain_expertise} 

    Consider the following recent questions:
    {seed_queries}

    Generate {n} *new* and *technically rich* questions, phrased naturally,without adding commentary or explanations 
    suitable for querying arXiv , NVIDIA developer , data science , machine learning , deep learning , python , 
    numpy, panda, scikit-learn  blogs. Avoid duplication, stay domain-relevant, and focus on "how" or "why" style 
    questions."""

    try:
        response = llm(prompt)
        raw_lines = response.splitlines()
        clean_queries = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            # keep only actual questions (start with number/bullet/** and end with '?')
            if "?" in line and not line.lower().startswith(
                    ("this question", "it explores", "the study", "this research")):
                # remove leading numbering or markdown artifacts
                q = re.sub(r"^[\d\.\*\-\s]+", "", line).strip()
                clean_queries.append(q)

        # deduplicate and log
        new_queries = list(dict.fromkeys(clean_queries))
        #new_queries = [q.strip("•- \n") for q in response.split("\n") if len(q.strip()) > 15]
        print(f"\n🧠 Generated {len(new_queries)} domain-specific queries from LLM.")
        with open(query_dir, "a") as f:
            for q in new_queries:
                f.write(f"{datetime.now().isoformat()} | {q}\n")
        return new_queries[:n]

    except Exception as e:
        print(f"⚠️ LLM query generation failed: {e}")
        return []


def determine_lang(code):
    if "__global__" in code or ".reg" in code or ".entry" in code:
        return "cuda"
    if "#include" in code:
        return "cpp"
    if "def " in code or "import " in code:
        return "python"
    return ""


def run_autonomous_cycle():
    print(f"\n=== 🚀 Autonomous Research Cycle Started ({datetime.now()}) ===\n")

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(RUNS_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    USER_QUERY_LOG = os.path.join(run_dir, "user_queries.log")
    GEN_QUERY_LOG = os.path.join(run_dir, "generated_queries.log")
    QUERY_LOG = os.path.join(run_dir, "queries.log")
    DIGEST_DIR = os.path.join(run_dir, "digests")
    os.makedirs(DIGEST_DIR, exist_ok=True)

    # 1️⃣ Fetch latest arXiv papers (only new ones)
    new_papers = fetch_new_papers()
    if new_papers:
        print(f"📚 Found {len(new_papers)} new research papers.")
    else:
        print("📭 No new papers detected in this cycle.")

    # 2️⃣ Update vector DB
    vectordb = build_or_update_vector_db(DOMAIN)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    try:
        llm = Ollama(model="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434")
    except Exception as e:
        print(f"⚠️ LLM initialization failed: {e}")
        return
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 3️⃣ Prepare dynamic queries
    base_queries = load_dynamic_queries()
    random.shuffle(base_queries)

    # pick a random subset each run
    num_to_select = random.randint(5, 12)  # choose 5–12 random queries each cycle
    selected_queries = base_queries[:num_to_select]

    # add a few generated ones too, for freshness
    generated_queries = generate_domain_queries(random.sample(base_queries, min(10, len(base_queries))), GEN_QUERY_LOG)
    # combine and deduplicate
    queries = list(dict.fromkeys(selected_queries + generated_queries))
    print(f"🧩 Selected {len(queries)} queries for this cycle (randomized subset).")

    with open(QUERY_LOG, "a") as f:
        f.write(f"number of queries : {len(queries)}")
        for q in queries:
            f.write(f"{datetime.now().isoformat()} | {q}\n")

    findings = {}
    for q in queries:
        tags = classify_domain(q)
        print(f"\n❓ [{', '.join(tags)}] {q}")

        context_docs = retriever.get_relevant_documents(q)
        context_text = "\n\n".join([d.page_content for d in context_docs[:5]])

        enhanced_prompt = f"""
        You are an expert in GPU systems, compiler optimization, and AI model acceleration.

        Context from research documents:
        {context_text}

        Using both this context and your own expertise,
        answer the following question thoroughly and with technical accuracy:

        {q}
        """
        # 🧠 DEBUG: Print the exact text sent to the LLM
        # print("\n=======================[LLM PROMPT DEBUG]=======================")
        # print(enhanced_prompt)
        # print("===============================================================\n")
        try:
            show_top_hits(q, vectordb, run_dir)
            ans = qa.invoke(enhanced_prompt)["result"]
            common_phrases = [
                "i don't know",
                "apologize",
                "no specific domain or topic",
                "I cannot provide",
                "I don't have enough",
                "does not mention "
            ]
            # 🧩 Check if the LLM gave a weak or uninformative answer
            if any(phrase.lower() in ans.lower() for phrase in common_phrases) or len(ans.split()) < 50:
                print("⚠️ Context insufficient — retrying with general knowledge (no context)...")
                try:
                    ans = llm.invoke(q)["message"] if isinstance(llm.invoke(q), dict) else llm.invoke(q)
                except Exception as e:
                    print(f"⚠️ Fallback failed: {e}")

            wrapped = "\n".join(textwrap.wrap(ans, width=WRAP_WIDTH))

            print(f"\n🤖 {wrapped}\n")
            for tag in tags:
                findings.setdefault(tag, []).append(f"{q} → {ans}")
        except Exception as e:
            print(f"⚠️ Error on '{q}': {e}")

        try:
            code_prompt = f"""
            Write a concise, well-commented example program in {random.choice(["Python", "C++", "CUDA PTX"])}
            that demonstrates the core concept discussed in the following answer:

            ---
            {ans}
            ---

            Focus on clarity and brevity (under 40 lines). Include inline comments to explain key steps.
            """
            code_example = llm.invoke(code_prompt)
        except Exception as e:
            code_example = f"[Code example unavailable: {e}]"

    # 4️⃣ Write summaries
    for tag, items in findings.items():
        folder = os.path.join(DIGEST_DIR, tag)
        os.makedirs(folder, exist_ok=True)
        print(f"\n🧩 Writing digests for tag: {tag}")

        for item in items:
            query_part, answer_part = item.split("→", 1) if "→" in item else (item, "")
            safe_query = "_".join(query_part.strip()[:80].split())
            filename = os.path.join(
                folder,
                f"digest_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            )

            # 🔍 Try to get top hits (retrieved context paragraphs)
            context_paragraphs = []
            try:
                if "retriever" in globals():
                    results = retriever.get_relevant_documents(query_part)
                    for doc in results[:5]:
                        context_paragraphs.append(
                            f"**{os.path.basename(doc.metadata.get('source', 'unknown'))}** → "
                            f"{textwrap.fill(doc.page_content[:600], width=WRAP_WIDTH)}"
                        )
            except Exception as e:
                context_paragraphs.append(f"[Context unavailable: {e}]")

            formatted_answer = format_answer(answer_part)

            digest_content = (
                f"# 🧩 Research Digest\n"
                f"## ❓ Query\n{query_part.strip()}\n\n"
                f"### 🧠 Summary\n{formatted_answer}\n\n"
            )

            digest_content += (
                "### 💻 Sample Implementation\n"
                f"```{determine_lang(code_example)}\n{code_example.strip()}\n```\n\n"
            )

            if context_paragraphs:
                digest_content += "### 📚 Retrieved Context\n" + "\n\n".join(context_paragraphs) + "\n\n"

            digest_content += f"---\nGenerated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

            with open(filename, "w") as f:
                f.write(digest_content)

            print(f"  📝 Saved digest for query: {query_part[:60]} → {os.path.basename(filename)}")

    with open(os.path.join(run_dir, "global_digest_index.md"), "a") as index:
        for tag, items in findings.items():
            for item in items:
                query = item.split("→", 1)[0].strip()
                index.write(f"- [{query}](./digests/{tag}/digest_{'_'.join(query[:80].split())}.md)\n")
    print("⏳ Sleeping until next research cycle...\n")


if __name__ == "__main__":
    while True:
        run_autonomous_cycle()
        time.sleep(REFRESH_INTERVAL * 60)
