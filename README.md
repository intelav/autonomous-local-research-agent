# Autonomous Local Research Agent

An autonomous, fully local research agent that:
- Continuously fetches relevant arXiv papers
- Builds and incrementally updates a vector database
- Uses local LLMs (Ollama) for retrieval-augmented reasoning
- Generates research summaries, insights, and code examples
- Runs fully offline after ingestion

## Features
- Incremental PDF embedding with fault tolerance
- Local RAG using Chroma + SentenceTransformers
- Ollama-based LLM inference (no cloud dependency)
- Autonomous research cycles
- arXiv ingestion with deduplication
- Resilient to bad PDFs and LLM outages

## Requirements
- Python 3.10+
- Ollama
- NVIDIA GPU recommended (works on CPU)

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3:8b-instruct
python run_agent.py
