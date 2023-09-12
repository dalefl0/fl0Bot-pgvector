import os
import re
import threading
from typing import Optional, Dict

from flask import Flask, request, jsonify
import psycopg2
import pandas as pd
import tiktoken
from dotenv import load_dotenv
import requests

from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pgvector.psycopg2 import register_vector

load_dotenv()

app = Flask(__name__)

host = os.getenv('PGHOST')
port = os.getenv('PGPORT')
user = os.getenv('PGUSER')
password = os.getenv('PGPASSWORD')
dbname = os.getenv('PGDATABASE')
endpoint = os.getenv('PGENDPOINT')

db = None  # Global variable to store the embeddings database

CONNECTION_STRING = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require&options=endpoint%3D{endpoint}"

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    if not string:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=103)

@app.route('/api/setup', methods=['PATCH'])
def setup_pgvector():
    connection = psycopg2.connect(CONNECTION_STRING)

    try:
        with connection:
            cur = connection.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        connection.commit()
        
        register_vector(connection)
        return jsonify({'response': 'pgvector extension installed successfully.'})

    except Exception as e:
        print(f"Error installing pgvector extension: {e}")
        raise

@app.route('/api/createembeddings', methods=['PATCH'])
def create_embeddings():
    global db
    df = pd.read_csv('training_data.csv')
    new_list = []
    for i in range(len(df.index)):
        text = df['content'][i]
        token_len = num_tokens_from_string(text)
        if token_len <= 512:
            new_list.append([df['workspaces'][i], df['content'][i], df['references'][i]])
        else:
            split_text = text_splitter.split_text(text)
            for j in range(len(split_text)):
                new_list.append([df['workspaces'][i], split_text[j], df['references'][i]])

    df_new = pd.DataFrame(new_list, columns=['workspaces', 'content', 'references'])

    loader = DataFrameLoader(df_new, page_content_column='content')
    docs = loader.load()

    embeddings = OpenAIEmbeddings()

    db = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="training_data",
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STRING
    )

    return jsonify({'response': 'embeddings succesfully created.'})

@app.route('/api/qa', methods=['POST'])
def qa(query: Optional[Dict] = None):
    global db

    if(query is None):
        data = request.get_json()
        query = data['query']

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k', openai_api_key=os.getenv('OPENAI_API_KEY'))
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    response = qa_stuff.run(query)

    return response
    
@app.route('/slack/action-endpoint', methods=['POST'])
def slack_action_endpoint():
    data = request.json
    challenge = data.get('challenge')
    
    if challenge:
        return jsonify(challenge), 200
    else:
        try:
            event_type = data.get('event', {}).get('type', None)
            if event_type == "app_mention":
                thread = threading.Thread(target=handleAppMention, args=(data.get('event', {}).get('text', None),))
                thread.start()
                # handleAppMention( data.get('event', {}).get('text', None))
                return jsonify({"message": "Success"}), 200
            else:
                return jsonify({"message": "Bad Request"}), 400
        except Exception as e:
            print(f"Error processing Slack event: {e}")
            return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    app.run()
