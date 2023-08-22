from flask import Flask, request, jsonify

from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import psycopg2
from pgvector.psycopg2 import register_vector
import os
import re
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from typing import Optional, Dict
import requests

load_dotenv()

app = Flask(__name__)

host = os.getenv('PGHOST')
port = os.getenv('PGPORT')
user = os.getenv('PGUSER')
password = os.getenv('PGPASSWORD')
dbname = os.getenv('PGDATABASE')
endpoint = os.getenv('PGENDPOINT')

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

@app.route('/api/qa', methods=['POST'])
def qa(query: Optional[Dict] = None):
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

    if(query is None): 
        data = request.get_json()
        query = data['query']

    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k', openai_api_key=os.getenv('OPENAI_API_KEY'))
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    response = qa_stuff.run(query)
    
    return jsonify({'response': response})


def handleAppMention(msg):
    print('Msg 1',msg)
    mentionRegex = re.compile(r'<@[\w\d]+>') 
    msg = re.sub(mentionRegex, '', msg)
    print('Msg 2',msg)
    query = msg

    response = qa(query)

    print(response)

    try:
    
        # Send a message to Slack
        webhook_url = os.getenv('SLACK_WEBHOOK')
        requests.post(webhook_url, json={'text': response})

        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"ERROR: {e}")
        return "Failed to process chat"
    

@app.route('/slack/action-endpoint', methods=['POST'])
def slack_action_endpoint():
    data = request.json
    challenge = data.get('challenge')

    print(data)
    
    if challenge:
        print(data)
        return jsonify(challenge), 200
    else:
        try:
            # event_type = data['event']['type']
            event_type = data.get('event', {}).get('type', None)
            if event_type == "app_mention":
                response = handleAppMention( data.get('event', {}).get('text', None))
                return jsonify({"message": "Success"}), 200
            else:
                return jsonify({"message": "Bad Request"}), 400
        except Exception as e:
            print(f"Error processing Slack event: {e}")
            return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    app.run()
