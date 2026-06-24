import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DB_PATH = "faiss_index"


def create_vector_db():
    loader = CSVLoader(
        file_path="dataset/dataset.csv",
        source_column="prompt",
        encoding="utf-8-sig",
        csv_args={
            "delimiter": "\t"
        }
    )

    documents = loader.load()

    vectordb = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vectordb.save_local(VECTOR_DB_PATH)

    print("Knowledge Base Created Successfully!")


def get_qa_chain():

    vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate(
        template="""
You are a customer support assistant.

Use ONLY the information from the context.

If the answer is not found, reply:

I don't know.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


if __name__ == "__main__":
    create_vector_db()