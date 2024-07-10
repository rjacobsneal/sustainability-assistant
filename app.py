import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from translate import Translator
from pyairtable import Api
import pandas as pd
import os

load_dotenv()

# SETUP AIRTABLE CONNECTION
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = "apptH6HITY4bbWqxi"
TABLE_ID = "tbl9zIBcSGqjPHzd0"
TABLE_NAME = "Example Project"

api = Api(AIRTABLE_API_KEY)
table = api.table(BASE_ID, TABLE_ID)

# SETUP TRANSLATOR
translator = Translator(to_lang="en")

# EXTRACT RELEVANT INFORMATION FROM AIRTABLE, PREPARE FOR CSV CREATION
def fetch_data_from_airtable():
    records = table.all()
    documents = []
    for record in records:
        fields = record["fields"]
        documents.append(fields)
        # STRATEGY 1, NO SIMILARITY SEARCH OR EMBEDDINGS USED
        # project_name = fields.get("Project Name", "")
        # project_description = fields.get("Project Description", "")
        # project_location = fields.get("Project Location", "")
        # organization = fields.get("Organization", "")
        # organization_type = fields.get("Organization Type", "")
        # project_website = fields.get("Project Website", "")
        
        # text = f"Project Name: {project_name}, Organization: {organization}, Organization Type: {organization_type}, Project Description: {project_description}, Project Location: {project_location}, Project Website: {project_website} \n"
        # documents.append(text)
    return documents

# STRATEGY 1
# all_documents = fetch_data_from_airtable()

# CONVERT AIRTABLE DATA TO CSV
df = pd.DataFrame(fetch_data_from_airtable())
df.to_csv('naturatec.csv', index=False)
loader = CSVLoader(file_path="naturatec.csv")
documents = loader.load()

# VECTORIZE AIRTABLE DATA
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# PERFORM SIMILARITY SEARCH
def retrieve_info(query):
    translated_query = translator.translate(query)
    similar_response = db.similarity_search(translated_query, k=5)
    page_contents_array = [doc.page_content for doc in similar_response]
    # print(page_contents_array)
    return page_contents_array


# SETUP LLM CHAIN AND PROMPT
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = """
You are an environmental sustainability consultant chatbot.
You work with individuals, organizations, and governments who seek to understand how to make the most environmentally sustainable desicions within their field, and/or how to best contribute to existing initiatives within their field.
I will share a user's message with you and you will respond directly to the user based on the project information I have provided to you.
You will follow ALL of the rules below:

1/ Prioritize collaboration when possible. If the user's message relates to existing project(s), always share the project name and organization. 
Always describe the project details; explain how the project is accomplishing their environmental objectives; responses should be generally informative about environmental issues and sustainability practices. 
Always describe how the project relates to the user's orginal message. 
Always conclude with a URL to the Project Website so that the user can learn more.
Share UP TO 3 RELEVANT projects. 

2/ Never fabricate new projects, organizations, etc. You may elaborate upon existing information, but all responses should be verifiable. 

3/ If none of the existing projects that relate whatsoever to the general content in the user's message, acknowledge that, and identify potential areas for the creation of future projects that relate to the user's message.

4/ If the user's message seems unfinished or not at all related to the aforementioned uses of this chatbot, reinform them of this chatbot's purpose and reprompt them for a relevant message.

5/ The response should sound like natural language, but it should maintain an informational tone. 

Below is a message I received from the user:
{message}

Here is are the projects from our database that are the most related to the user's message:
{best_project}

Please write the best response to this user:
"""

prompt = PromptTemplate(input_variables=["message", "best_project"], template=template)

chain = prompt | llm


# RETRIEVAL AUGMENTED GENERATION
def generate_response(message):
    best_project = retrieve_info(message)
    # STRATEGY 1
    # best_project = all_documents
    response = chain.invoke({"message": message, "best_project": best_project})
    return response.content


# TESTING IN TERMINAL
customer_message = """"
I want to work with projects that deal with marine ecosystems. 
"""
response = generate_response(customer_message)
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
