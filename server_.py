import os
import pickle
from typing import Optional, Tuple
from threading import Lock

from flask import Flask, request, jsonify

from app.query_data import get_chain
# from app.ingest_data import create_vectorstore

app = Flask(__name__)

# Load the vectorstore
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# Check if the vectorstore file exists, and if not, create it

# def check():
#     vectorstore_file = "vectorstore.pkl"
    
#     with open(vectorstore_file, "rb") as f:
#         vectorstore = pickle.load(f)
#         return vectorstore

# class ChatWrapper:
#     def __init__(self):
#         self.lock = Lock()

#     def __call__(
#         self, inp: str, history: Optional[Tuple[str, str]], chain
#     ):
#         """Execute the chat functionality."""
#         self.lock.acquire()
#         try:
#             history = history or []
#             # If chain is None, that is because no API key was provided.
#             if chain is None:
#                 history.append((inp, "Bad OpenAI key"))
#                 return history, history

#             # # Run chain and append input.
#             # chain = get_chain(vectorstore)
#             # print(chain)

#             output = chain({"question": inp, "chat_history": history})["answer"]
#             print(output)

#             history.append((inp, output))
#         except Exception as e:
#             raise e
#         finally:
#             self.lock.release()
#         return history, history

# def chat_api():
#     data = request.json
#     inp = data.get("question")
#     history = []
#     # chain = get_chain(check())
#     chain=get_chain(vectorstore)
#     # print(chain)
#     if chain is None:
#         return jsonify({"error": "Invalid OpenAI API key"}), 400

#     chat = ChatWrapper()
#     output,_ = chat(inp, history, chain)

#     return {"output": output}


# @app.route("/create_vectorstore", methods=["POST"])
# def create_vectorstoree():

#     create_vectorstore()
#     # Return a response indicating success
#     return jsonify({"message": "Vector store created successfully!"}), 201

def chat_bot():
    data = request.json
    question = data.get("question")
    qa_chain = get_chain(vectorstore)
    chat_history = []

    result = qa_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))

    return {"ai": (result["answer"])}


app.add_url_rule("/api/chat", "chat_api", chat_api, methods=["POST"])
app.add_url_rule("/api/ai", "chat_bot", chat_bot, methods=["POST"])

if __name__ == "__main__":
    app.run()


#  myenv\Scripts\activate
# python ingest_data.py
# python app.py