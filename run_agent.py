from flask import Flask, render_template, request, jsonify
from agent_local_rag import get_qa_chain
import argparse

app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--domain", default="research", help="Choose knowledge base domain")
args = parser.parse_args()
qa = get_qa_chain(domain=args.domain)


@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("query", "")
    if not question:
        return jsonify({"response": "Please ask a question."})
    answer = qa.run(question)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
