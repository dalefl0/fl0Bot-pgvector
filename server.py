from flask import Flask, request, jsonify
from langchain.vectorstores.pgvector import PGVector
import os
from dotenv import load_dotenv
import pandas as pd
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import psycopg2 as dbb
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import urllib.parse

app = Flask(__name__)

# Load environment variables and create the connection string
load_dotenv()

host = os.getenv('HOST')
port = os.getenv('DBPORT')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
dbname = os.getenv('DB')
endpoint = os.getenv('ENDPOINT')
encoded_endpoint = urllib.parse.quote(endpoint)
sslmode = 'require'
# sslmode = 'disable'

CONNECTION_STRING = f"host={host} dbname={dbname} user={user} password=endpoint={endpoint};{password} sslmode={sslmode}"
# CONNECTION_STRING_NO_DB = f"host={host} dbname={dbname} user={user} password={password} sslmode={sslmode} options='-c search_path=public -c options=endpoint={endpoint}'"


# CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require&options=endpoint%3D{endpoint}"
# CONNECTION_STRING_NO_DB = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"


# def database_exists():
#     # Connect to PostgreSQL without specifying a database
#     connection = dbb.connect(CONNECTION_STRING_NO_DB)

#     # Check if the database exists
#     cur = connection.cursor()
#     cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
#     return cur.fetchone() is not None

# def create_db():
#     if not database_exists():

#         # Connect to PostgreSQL without specifying a database
#         connection = dbb.connect(CONNECTION_STRING_NO_DB)

#         try:

#             # Set the isolation level to autocommit to avoid running inside a transaction
#             connection.set_isolation_level(dbb.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

#             # Create the database if it does not exist
#             cur = connection.cursor()
#             cur.execute(f"CREATE DATABASE {dbname}")

#         except Exception as e:
#             print(f"Error creating database '{dbname}': {e}")
#             raise
#         finally:
#             # Close the connection after creating the database
#             if connection is not None:
#                 connection.close()

# create_db()


# def user_exists():
#     # Connect to PostgreSQL without specifying a database
#     connection = dbb.connect(CONNECTION_STRING_NO_DB)

#     # Check if the database exists
#     cur = connection.cursor()
#     cur.execute(f"SELECT 1 FROM pg_user WHERE usename = '{user}'")
#     return cur.fetchone() is not None

# def create_user():
#     if not user_exists():

#         # Connect to PostgreSQL without specifying a database
#         connection = dbb.connect(CONNECTION_STRING_NO_DB)

#         try:

#             # Set the isolation level to autocommit to avoid running inside a transaction
#             connection.set_isolation_level(dbb.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

#             # Create the database if it does not exist
#             cur = connection.cursor()
#             cur.execute(f"CREATE USER {user} WITH PASSWORD '{password}'")

#         except Exception as e:
#             print(f"Error creating user '{user_exists}': {e}")
#             raise
#         finally:
#             # Close the connection after creating the database
#             if connection is not None:
#                 connection.close()

# create_user()

# Automate the installation of pgvector extension and table setup
def setup_pgvector():
    connection = dbb.connect(CONNECTION_STRING)

    try:
        # Connect to PostgreSQL database and create the extension
        with connection:
            cur = connection.cursor()

            #install pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        connection.commit()
        
        from pgvector.psycopg2 import register_vector
        register_vector(connection)
        print("pgvector extension installed successfully.")

    except Exception as e:
        print(f"Error installing pgvector extension: {e}")
        raise
        

setup_pgvector()


# Load the CSV data and preprocess it
df = pd.read_csv('app/social_media.csv')

# Helper function: calculate number of tokens
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Split text into chunks of 512 tokens, with 20% token overlap
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=103)

# Create a new list by splitting up text into token sizes of around 512 tokens
new_list = []
for i in range(len(df.index)):
    text = df['content'][i]
    token_len = num_tokens_from_string(text)
    if token_len <= 512:
        new_list.append([df['title'][i], df['content'][i], df['origin'][i]])
    else:
        # split text into chunks using text splitter
        split_text = text_splitter.split_text(text)
        for j in range(len(split_text)):
            new_list.append([df['title'][i], split_text[j], df['origin'][i]])

df_new = pd.DataFrame(new_list, columns=['title', 'content', 'origin'])

# Load documents from Pandas dataframe for insertion into database
loader = DataFrameLoader(df_new, page_content_column='content')
docs = loader.load()

# Create OpenAI embedding using LangChain's OpenAIEmbeddings class
embeddings = OpenAIEmbeddings()
# query_string = "PostgreSQL is my favorite database"
# embed = embeddings.embed_query(query_string)

# Create a PGVector instance to house the documents and embeddings
db = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="blog_posts",
    distance_strategy=DistanceStrategy.COSINE,
    connection_string=CONNECTION_STRING
)

# @app.route('/api/embed', methods=['POST'])
# def embed_text():
#     data = request.get_json()
#     query_string = data['query']
    
#     # Initialize embeddings and create the OpenAI embedding for the query
#     embeddings = OpenAIEmbeddings()
#     embed = embeddings.embed_query(query_string)
    
#     return jsonify(embed)

# @app.route('/api/similarity_search', methods=['POST'])
# def similarity_search():
#     data = request.get_json()
#     query = data['query']
    
#     # Fetch the k=3 most similar documents
#     docs = db.similarity_search(query, k=3)
    
#     # Prepare the response data
#     response_data = []
#     for doc in docs:
#         doc_content = doc.page_content
#         doc_metadata = doc.metadata
#         response_data.append({
#             'content': doc_content,
#             'title': doc_metadata['title'],
#             'url': doc_metadata['url']
#         })
    
#     return jsonify(response_data)

@app.route('/api/qa', methods=['POST'])
def qa():
    data = request.get_json()
    query = data['query']
    
    # Create the retriever from the database with k=3 results
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Initialize the language model and QA pipeline
    llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k', openai_api_key=os.getenv('OPENAI_API_KEY'))
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # Run the QA pipeline with the given query
    response = qa_stuff.run(query)
    
    return jsonify({'response': response})

@app.route('/api/ai', methods=['POST'])
def chat_bot():
    data = request.json
    question = data.get("question")
    ai = ''
    return {"ai": ai}

if __name__ == "__main__":
    app.run()
