from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import uvicorn
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
load_dotenv()

db = FAISS.load_local("faiss_index", embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
model = ChatGroq(model_name='llama-3.3-70b-versatile')
retriever = db.as_retriever(search_type='similarity',search_kwargs={"k":4})

system_prompt = (
    "You are a PDF Reader Assistant."
    "Provide concise and professional responses."
    "If the user asks a part of PDF, give clear and helpful information about it."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate(
    [
        ("system",system_prompt),
        ("human", "{question}"),
    ]
    )

context_chain = ({
    "context": itemgetter("question") | retriever,
    "question": itemgetter('question'),
}
| prompt | model | StrOutputParser())

app = FastAPI(
    title="Simbolo ChatBot API",
    version="1.0",
    description="Simbolo chatbot by using Langchain"
)

add_routes(app, context_chain, path ="/coding")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
