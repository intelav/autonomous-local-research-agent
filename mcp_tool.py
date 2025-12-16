# mcp_tools.py
from openai_mcp import MCPTool

tool = MCPTool(name="research_agent")

@tool.function
def fetch_new_papers(topic: str) -> str:
    from autonomous_research_agent import fetch_new_papers
    papers = fetch_new_papers(topic)
    return f"Fetched {len(papers)} papers for topic: {topic}"

@tool.function
def query_vector_db(domain: str, question: str) -> str:
    from vector_manager import build_or_update_vector_db
    vectordb = build_or_update_vector_db(domain)
    retriever = vectordb.as_retriever()
    results = retriever.get_relevant_documents(question)
    return "\n".join([r.page_content[:300] for r in results])

if __name__ == "__main__":
    tool.serve()
