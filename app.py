# app.py
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from openai import OpenAI
import chromadb
import os

# -------- Initialization --------
app = Flask(__name__)
CORS(app)
app.secret_key = "24689"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("mmk_site")

# -------- Agent function --------
def answer_question(query, chat_history=None):
    chat_history = chat_history or []

    # Retrieve relevant website/profile chunks
    results = collection.query(query_texts=[query], n_results=3)
    context = " ".join(results["documents"][0]) if results["documents"] else ""

    # System prompt for intelligence and friendliness
    system_prompt = """
You are MMK Agent, a smart and friendly assistant for the mmk website and Kgaugelo Mmakola's profile.
- Answer using website content or profile info when possible.
- Greet users naturally.
- Provide contact info (email, GitHub, LinkedIn) if asked.
- If info is missing, suggest visiting the website.
- Be helpful and conversational.
"""

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Relevant info:\n{context}"})
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

# -------- Flask route --------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    # Keep session chat history
    chat_history = session.get("chat_history", [])
    bot_reply = answer_question(user_message, chat_history)

    # Update history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": bot_reply})
    session["chat_history"] = chat_history

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
