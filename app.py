from prompt_instructions import prompt_instructions
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from translate import Translator
from pyairtable import Api
import streamlit as st
import pandas as pd
import os

load_dotenv()

# SETUP AIRTABLE CONNECTION
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = "apptH6HITY4bbWqxi"
PROJECT_TABLE_ID = "tbl9zIBcSGqjPHzd0"

api = Api(AIRTABLE_API_KEY)
project_table = api.table(BASE_ID, PROJECT_TABLE_ID)

# SETUP TRANSLATOR
translator = Translator(to_lang="en")

# EXTRACT RELEVANT INFORMATION FROM AIRTABLE, PREPARE FOR CSV CREATION
def fetch_data_from_airtable(table):
    records = table.all()
    documents = []
    for record in records:
        fields = record["fields"]
        documents.append(fields)
    return documents

# CONVERT AIRTABLE DATA TO CSV
def airtable_to_csv(table, table_name):
    df = pd.DataFrame(fetch_data_from_airtable(table))
    table_name = f'{table_name}.csv'
    df.to_csv(table_name, index=False)
    loader = CSVLoader(file_path=table_name)
    documents = loader.load()
    return documents

project_csv = airtable_to_csv(project_table, "project")

# VECTORIZE AIRTABLE DATA
embeddings = OpenAIEmbeddings()
project_db = FAISS.from_documents(project_csv, embeddings)

# PERFORM SIMILARITY SEARCH
def retrieve_info(db, k, message):
    translated_message = translator.translate(message)
    similar_response = db.similarity_search(translated_message, k=k)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


# SETUP LLM CHAIN AND PROMPT
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = f"""
{prompt_instructions}

Below is a message I received from the user:
{{message}}

Please write the best response to this user:
"""

prompt = PromptTemplate(input_variables=["message", "best_project"], template=template)

chain = prompt | llm


# RETRIEVAL AUGMENTED GENERATION
def generate_response(message):
    best_project = retrieve_info(project_db, 5, message)
    response = chain.invoke({"message": message, "best_project": best_project})
    return response.content


# TESTING IN TERMINAL
user_message = """"
I want to work with a project in Latin America. 
"""
response = generate_response(user_message)
print(response)


# BUILD APP WITH STREAMLIT
def main():
    st.set_page_config(page_title="sustainability assistant", page_icon=":seedling:")

    st.header("sustainability assistant :seedling:")
    message = st.text_area("type your message")

    if message:
        st.write("generating response...")

        result = generate_response(message)

        st.info(result)


if __name__ == "__main__":
    main()
