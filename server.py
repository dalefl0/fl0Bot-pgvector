from flask import Flask, request, jsonify

from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import psycopg2
import csv
from pgvector.psycopg2 import register_vector
import os
import pandas as pd
import tiktoken
from dotenv import load_dotenv

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

@app.route('/api/setup', methods=['POST'])
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

@app.route('/api/csv', methods=['POST'])
def to_csv():
    

    data = [
        {"Serial No": 1, "Title": "Facebook", "Content": "Social networking platform", "Country Origin": "United States"},
        {"Serial No": 2, "Title": "Twitter", "Content": "Microblogging platform", "Country Origin": "United States"},
        {"Serial No": 3, "Title": "Instagram", "Content": "Photo and video sharing platform", "Country Origin": "United States"},
        {"Serial No": 4, "Title": "LinkedIn", "Content": "Professional networking platform", "Country Origin": "United States"},
        {"Serial No": 5, "Title": "YouTube", "Content": "Video-sharing platform", "Country Origin": "United States"},
        {"Serial No": 6, "Title": "TikTok", "Content": "Short-form video platform", "Country Origin": "China"},
        {"Serial No": 7, "Title": "WhatsApp", "Content": "Messaging platform", "Country Origin": "United States"},
        {"Serial No": 8, "Title": "Snapchat", "Content": "Multimedia messaging app", "Country Origin": "United States"},
        {"Serial No": 9, "Title": "WeChat", "Content": "Social media and messaging app", "Country Origin": "China"},
        {"Serial No": 10, "Title": "Reddit", "Content": "Social news aggregation platform", "Country Origin": "United States"},
    ]

    # Define the CSV file path
    csv_file = "social_media.csv"

    # Write data to the CSV file
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["Sno", "title", "content", "origin"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerows(data)
        
    return jsonify({'response': 'CSV file created successfully.'})

@app.route('/api/qa', methods=['POST'])
def qa():
    df = pd.read_csv('social_media.csv')
    new_list = []
    for i in range(len(df.index)):
        text = df['content'][i]
        token_len = num_tokens_from_string(text)
        if token_len <= 512:
            new_list.append([df['title'][i], df['content'][i], df['origin'][i]])
        else:
            split_text = text_splitter.split_text(text)
            for j in range(len(split_text)):
                new_list.append([df['title'][i], split_text[j], df['origin'][i]])

    df_new = pd.DataFrame(new_list, columns=['title', 'content', 'origin'])

    loader = DataFrameLoader(df_new, page_content_column='content')
    docs = loader.load()

    embeddings = OpenAIEmbeddings()

    db = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="social_media",
        distance_strategy=DistanceStrategy.COSINE,
        connection_string=CONNECTION_STRING
    )

    data = request.get_json()
    query = data['query']

    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k', openai_api_key=os.getenv('OPENAI_API_KEY'))
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    response = qa_stuff.run(query)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run()
