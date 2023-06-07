import csv
import os
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import Document


def read_csv_into_vector_doc(file, text_cols):
    with open(file, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        text_data = []
        for row in csv_reader:
            text = ' '.join([row[col] for col in text_cols])
            text_data.append(text)
        return [Document(page_content=text) for text in text_data]


openai.api_key = os.environ["OPENAI_API_KEY"]

if not openai.api_key:
    print('OpenAI API key not found in environment variables.')
    exit()

data = read_csv_into_vector_doc("new_nsu_data.csv",
                                     ["Function", "Building", "Office", "Office Number", "Additional information", ])
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectors = FAISS.from_documents(data, embeddings)
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai.api_key),
    retriever=vectors.as_retriever())
history = []

while True:
    query = input("Enter Your Query: ")
    print(chain({"question": query, "chat_history": history})["answer"])
