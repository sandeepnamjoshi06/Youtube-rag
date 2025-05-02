from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pickle
import os
load_dotenv()
pinecone_api_key=os.getenv("PINECONE_API_KEY")
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
def retrivers():
     loader = DirectoryLoader("data",
                         glob='*.txt',
                         loader_cls=TextLoader)
     docs=loader.load()
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
     doc = text_splitter.split_documents(docs)
     print("docs---",doc)
     print("chunk made successfully")
     model_name = "sentence-transformers/all-mpnet-base-v2"
     model_kwargs = {'device': 'cpu'}
     encode_kwargs = {'normalize_embeddings': False}
     hf = HuggingFaceEmbeddings(
          model_name=model_name,
          model_kwargs=model_kwargs,
          encode_kwargs=encode_kwargs
          )

     embeddings=hf
     index_name="youtube"
     pc = Pinecone(api_key="pcsk_2uHYQ8_HW9jhYmPDytSqSP5w1zU3c9UALQ4jkaDd8pWSXCmtRQx5A4hQTDnA3GFBdGGP6c")
     if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
     else:
        print("Delete index and create index--->")
        pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=768,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
     pinecone_index = pc.Index(index_name)
     print("pinecone_index-----",pinecone_index)
     PineconeVectorStore.from_documents(doc, embeddings, index_name=index_name)
     keyword_retriever=BM25Retriever.from_documents(doc)
     with open("bm25_retriever.pkl", "wb") as f:
            pickle.dump(keyword_retriever, f)
     print("pinecone index create--------------", pinecone_index)
     print("Resources loaded and indexed.and embedding file save successfully")

     return "retriver made successfully"