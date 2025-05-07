from langchain_groq import ChatGroq
from prompt import prompt,parser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
import os
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from utility import *
from prompt import prompt,parser
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
pinecone_api_key=os.getenv("PINECONE_API_KEY")
llm=ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    )
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
          model_name=model_name,
          model_kwargs=model_kwargs,
          encode_kwargs=encode_kwargs
          )
def ask_question(question:str):
    embeddings=hf
    docsearch = PineconeVectorStore.from_existing_index("youtube", embeddings)
    vec_retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 3,'lambda_mult': 0.6})
    with open("bm25_retriever.pkl", "rb") as f:
         keyword_retriever = pickle.load(f)
    retriever=EnsembleRetriever(retrievers=[vec_retriever,keyword_retriever],weights=[0.3,0.7])
    parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
    print(parallel_chain.invoke(question))
    main_chain = parallel_chain|prompt|llm|parser
    response=main_chain.invoke(question)
    print("response-----",response)
    return response
