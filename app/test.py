from langchain.vectorstores.pgvector import PGVector
import os


from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

host= os.getenv('HOST')
port= os.getenv('DBPORT')
user= os.getenv('USER')
password= os.getenv('PASSWORD')
dbname= os.getenv('DB')


CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=disable"

import pandas as pd
import numpy as np
df = pd.read_csv('app/social_media.csv')
df.head()


import tiktoken
from langchain.text_splitter import TokenTextSplitter
# Split text into chunks of 512 tokens, with 20% token overlap
text_splitter = TokenTextSplitter(chunk_size=512,chunk_overlap=103)


# Helper func: calculate number of tokens
def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#list for smaller chunked text and metadata
new_list = []

# Create a new list by splitting up text into token sizes of around 512 tokens
for i in range(len(df.index)):
    text = df['content'][i]
    token_len = num_tokens_from_string(text)
    if token_len <= 512:
        new_list.append([df['title'][i],
        df['content'][i], 
        df['origin'][i]])
    else:
        #split text into chunks using text splitter
        split_text = text_splitter.split_text(text)
        for j in range(len(split_text)):
            new_list.append([df['title'][i],
            split_text[j],
            df['origin'][i]])


df_new = pd.DataFrame(new_list, columns=['title', 'content', 'origin'])
df_new.head()



#load documents from Pandas dataframe for insertion into database
from langchain.document_loaders import DataFrameLoader

# page_content_column is the column name in the dataframe to create embeddings for
loader = DataFrameLoader(df_new, page_content_column = 'content')
docs = loader.load()


from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# Create OpenAI embedding using LangChain's OpenAIEmbeddings class
query_string = "PostgreSQL is my favorite database"
embed = embeddings.embed_query(query_string)
print(len(embed)) # Should be 1536, the dimensionality of OpenAI embeddings
print(embed[:5]) # Should be a list of floats



# Create a PGVector instance to house the documents and embeddings
from langchain.vectorstores.pgvector import DistanceStrategy
db = PGVector.from_documents(
    documents= docs,    
    embedding = embeddings,
    collection_name= "social_media",
    distance_strategy = DistanceStrategy.COSINE,
    connection_string=CONNECTION_STRING)


from langchain.schema import Document

# Query for which we want to find semantically similar documents
query = "Tell me about a video sharing platform with yellow icon"

#Fetch the k=3 most similar documents
docs =  db.similarity_search(query, k=3)


# Interact with a document returned from the similarity search on pgvector
doc = docs[0]

# Access the document's content
doc_content = doc.page_content
# Access the document's metadata object
doc_metadata = doc.metadata

print("Content snippet:" + doc_content[:500])
print("Document title: " + doc_metadata['title'])
print("Document Origin: " + doc_metadata['origin'])



# Create retriever from database
# We specify the number of results we want to retrieve (k=3)
retriever = db.as_retriever(
    search_kwargs={"k": 3}
    )



from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature = 0.0, model = 'gpt-3.5-turbo-16k')



from langchain.chains import RetrievalQA
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    verbose=True,
)



query =  "Tell me about a video sharing platform with yellow icon"

response = qa_stuff.run(query)

print(response)
