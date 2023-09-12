---
title: Building a Super Powered Slack Bot with Langchain, GPT API, FLask, and FL0
date: 2023-08-20
Time to read: 10 mins
tags: engineering, machine-learning, fl0, chat-gpt
---

# Building a Super Powered Slack Bot with Langchain, GPT API, FLask, and FL0

## TL;DR

By the end of this guide, we'll have a dynamic Slack bot that can answer questions about the data our model was trained on using LangChain. Additionally, by integrating with the GPT API, we're essentially equipping our chatbot with superpowers for more comprehensive and informed responses. 🤖✨.

## Introduction

"With the emergence of GPT, LLMs, and other AI/ML technologies, many developers have effortlessly crafted advanced chatbots 🧑‍💻.

We've already created a [Chatbot](https://www.fl0.com/blog/building-a-slack-chatbot-with-gpt-api-nodejs-and-fl0) using the GPT API, NodeJs, and FL0. While it previously depended solely on GPT's capabilities to answer questions, we're about to take it a notch higher. By integrating with LangChain, our chatbot will not only tap into GPT's vast knowledge but also utilize our specific data.

In this guide, we'll set up a Slack chatbot named FL0Bot_v2. This chatbot will address questions related to the specific data our model has been trained on, leveraging GPT's powerful capabilities. 💬

Our technology stack features Flask for the backend, with the Postgres extension, pgvector, handling embedding storage. Plus, we'll be deploying our application smoothly with the help of FL0 🚀.

Eager to get started? Before diving in, let's kick things off with a bit of humor. Enjoy this xkcd comic strip to lighten the mood 👇."

## Getting Started

Let's kick off the construction of our exceptional chatbot 💬.

For efficiency, we'll employ the “fl0zone/template-python” template in this guide.

In this template we have our basic python-flask application and postgres database.

## Folder Structure

At the completion of the tutorial, the folder structure would look like this 👇

```
fl0-bot-v2/
├── Procfile
├── myenv
├── server.py
├── training_data.csv
└── requirements.txt
```

And here’s a high level overview of what we are gonna build 👀

//diagram

## Step 1: Project Setup

After we have created our new project using the above template, we would first need to install a few packages.

We would update the `requirements.txt` file with the packages that we require 👇

`requirements.txt`

```txt
Flask
gunicorn
langchain
openai
python-dotenv
tiktoken
pgvector
pandas
numpy
psycopg2
```
Now back to the tutorial.

## Step 2: Project Configurations

Let's start by creating a `.env` file to house our environment variables 👇:

`.env`

```txt
OPENAI_API_KEY = "our open-ai api key"
PGHOST="postgres-db-host"
PGDATABASE="postgres-db-name"
PGUSER="postgres-db-user"
PGPASSWORD="postgres-db-password"
PGPORT="postgres-db-port"
PGENDPOINT="postgres-db-endpoint"
SLACK_WEBHOOK="slack-channel-webhook"
```

These variables will be imported and utilized within the server file located at /server.py.


Now let's update the `Procfile` file to better accommodate our application's needs 👇:

`Procfile`

```txt
web: gunicorn server:app \
   --workers 1 \
   --worker-class uvicorn.workers.UvicornWorker \
   --bind 0.0.0.0:5000 \
   --timeout 5000
```

We've got a Procfile here, which is typical for deploying web applications. In this configuration, we're booting up a Gunicorn server that's set to run our Python web application from our server module. We're leveraging Uvicorn workers, which are primed for handling async ASGI apps. Our app's going to listen on port 5000. With just one worker and a long wait time of 5000 seconds, we might need to change those numbers based on how busy our app gets.


`training_data.csv`

For our model's training, we've used a file named `training_data.csv`. This file contains info about FL0 platform's workspaces. Think of it as the model's study material. Even though datasets can be really big, we're just using this for our example.

## Step 3: Setting up our application

Now, we will update the file named `server.py` and add our [NeonDB](https://neon.tech/) PostgreSQL database connection and code for our APIs.

---

In this file we would need to create the following routes:

`PATCH /api/setup` - Basic Database Setup 
`POST /api/qa` - Chat API b/w User and Server
`POST /slack/action-endpoint` - Handles Slack Events

---

So let's get started 👇

`server.py`

```python
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
    
    return response


def handleAppMention(msg):
    mentionRegex = re.compile(r'<@[\w\d]+>') 
    msg = re.sub(mentionRegex, '', msg)
    query = msg

    response = qa(query)

    try:
    
        # Send a message to Slack
        webhook_url = os.getenv('SLACK_WEBHOOK')
        requests.post(webhook_url, json={'text': response})
    except Exception as e:
        print(f"ERROR: {e}")
    

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
                return jsonify({"message": "Success"}), 200
            else:
                return jsonify({"message": "Bad Request"}), 400
        except Exception as e:
            print(f"Error processing Slack event: {e}")
            return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    app.run()

```

The provided code establishes a Flask application ⚡️ with three main endpoints: `/api/setup`, `/api/qa` and `/slack/action-endpoint`.

The endpoint `/api/setup` ensures the pgvector extension is integrated with our postgres database if it hasn't been already.

On the other hand, the `/api/qa` endpoint fetches, processes and returns the answer to the user's query.

And the last endpoint `/slack/action-endpoint` establishes the slack connection and process any incomming requests. Here, a new background thread is initilized to process Slack mentions, allowing the main application to respond quickly without waiting for the mention handling to complete. It's a method to achieve non-blocking behavior in our application.

Let's deep dive into the `/api/qa` endpoint and its associated components:
1. Reading the Training Data:
    - Using the pandas library, the code reads from training_data.csv, which presumably contains the training data that the chatbot uses for generating responses.
    - Dataframe df is created, which will hold this data.

2. Processing and Splitting Text:
    - An empty list new_list is initialized to store processed content.
    - For every row in df, it calculates the token length of the content using the num_tokens_from_string function.
    - If the token length of the content is less than or equal to 512, it is directly added to new_list.
    - If it exceeds 512, the content is split using text_splitter.split_text(text). The split content is then added to new_list in chunks.

3. Creating a New DataFrame:
    - A new DataFrame df_new is created using the new_list. This new DataFrame will have the columns: 'workspaces', 'content', and 'references'.

4. Loading Processed Data:
    - DataFrameLoader is utilized to load the processed content from df_new.
    - The loaded content is referred to as docs in the code.

5. Generating Embeddings:
    - The OpenAIEmbeddings() class is used to generate embeddings for the docs.

6. Storing Embeddings in Database:
    - Using the PGVector.from_documents function, the generated embeddings are stored into a Postgres database configured with pgvector. This allows for efficient retrieval based on vector similarity.
    - The content from the docs is embedded and stored in the database under the collection name "training_data".

7. Handling User Queries:
    - The function waits for a POST request containing a user query.
    - When a query is received, it is stored in the query variable.

8. Searching for Relevant Content:
    - Using the stored embeddings, the system searches for the most relevant content to the given query using the db.as_retriever method. It retrieves the top 3 (k=3) most relevant entries.

9. Generating a Response:
    - A language model (ChatOpenAI with the 'gpt-3.5-turbo-16k' model) is then used to generate a suitable response based on the retrieved content.
    - The RetrievalQA.from_chain_type method combines the powers of the retriever and the language model to generate this response.

10. Returning the Response:
    - Finally, the generated response is packaged into a JSON format and returned to the user.

To start the Flask server on the local server, we can run the following command:

```bash
python server.py
```

Now, upon using `http://localhost:5000/{api_endpoint}`, we should be able to use the apis.


## Step 4: Deploying with FL0

With our API and database set up and ready to go, the next step is launching them onto a server! 🚀

In this guide, we're leveraging FL0, a platform specifically tailored for the seamless deployment of Python Flask apps, complete with database integration.

All we need to do is upload our repository to `GitHub`.

From there, the deployment process involves simply "Linking our GitHub account" and choosing our project.

Afterward, we'll input the environment variables from our .env file.


## Step 5: Setting up Slack App

Now that our project is set up, let’s create our Slack App.

We would visit https://api.slack.com/apps and click on Create New App.

We would name our bot “FL0Bot” 😁

In the Event Subscriptions section, we would enable events, set the request URL, and subscribe to bot events: app_mention

We would also need to get our webhook and pass it as an environment variable to our FL0 hosting.


## Conclusion

"We've journeyed through the intricate process of crafting an enhanced chatbot, taking advantage of both GPT's extensive capabilities and our specific LangChain data. With the integration of Flask, Postgres, and OpenAI's GPT, and the ease of deployment via !!FL0!!, we now have a chatbot that's more responsive and informative than ever.

Explore the fruits of our labor by checking out the FL0Bot Repo ➡️ Visit FL0Bot Repo.

The synergy of OpenAI's robust APIs and !!FL0!!'s streamlined deployments makes the process of crafting top-notch AI bots virtually seamless 🚀🎉. Ready to embark on a similar journey? Visit fl0.com and kickstart the creation of your bespoke AI-powered bots 🧑‍💻."