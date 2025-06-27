import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

def setup():
  # Loading
  loader =  PyPDFLoader('iphone-16-info.pdf')
  pages = loader.load()

  # Splitting
  text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=500,
     chunk_overlap=20,
     length_function=len,
     is_separator_regex=False,
  )
  texts = text_splitter.split_documents(pages)

  # Embedding
  embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

  # Init Vector Database
  if os.path.exists("faiss_index"):
        db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
  else:
        db = FAISS.from_documents(texts, embeddings_model)
        db.save_local("faiss_index")

  # Similarity Search
  query = input("Ask a question about the doc: ")
  docs = db.similarity_search(query)
  return query,docs

def bot(query,docs):
  load_dotenv()
  api_key = os.getenv('GROQ_API_KEY')
  if not api_key:
    raise ValueError("GROQ_API_KEY not set")
  os.environ['GROQ_API_KEY'] = api_key

  llm = init_chat_model("llama3-8b-8192", model_provider="groq")
  response = llm.invoke(f"For the user query: {query}. Answer using the following knowledge: {docs}")
  print(response.content)

def main():
  query,docs = setup()
  bot(query,docs)

if __name__ == "__main__":
  main()