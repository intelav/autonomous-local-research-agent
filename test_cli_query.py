import textwrap
from datetime import datetime

from agent_local_rag import get_vector_db
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os


def show_top_hits(query):
    """Show top retrieved chunks with actual text content."""
    print("\n🔎 Top retrieved chunks:")
    docs_with_scores = vectordb.similarity_search_with_score(query, k=5)
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"\n  {i:02d}. [{score:.3f}] {source}")
        print("      └── Context:")
        print(textwrap.fill(snippet[:600], width=100))  # show up to 600 chars
    print("\n" + "-" * 120)


# Settings
WRAP_WIDTH = 100  # adjust for your terminal width (80–120 is good)

domain = "research"
vectordb = get_vector_db(domain)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

print(f"\n📚 Domain loaded: {domain}")
print("💬 Type your question (or 'exit' to quit)\n")

while True:
    query = input("\n❓ Ask: ")
    if query.lower() in ["exit", "quit"]:
        break

    show_top_hits(query)  # <--- new debug helper
    answer = qa.run(query)
    print("\n🤖 Answer:\n")
    print(textwrap.fill(answer, width=100))
    print("-" * 100)

    with open("runs/user_queries.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} | {query}\n")