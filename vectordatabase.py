from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os
import glob

def build_vectordb():
    docs = []

    # Load .txt files
    txt_loader = DirectoryLoader("supplychain_docs", glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    docs.extend(txt_docs)

    # Load all .csv files in the directory
    csv_files = glob.glob(os.path.join("supplychain_docs", "**/*.csv"), recursive=True)
    for csv_file in csv_files:
        csv_loader = CSVLoader(file_path=csv_file)
        csv_docs = csv_loader.load()
        docs.extend(csv_docs)

    print(f"Loaded {len(docs)} documents (including .txt and .csv files).")

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")

    if not split_docs:
        print("No content found to embed.")
        return

    # Create embeddings + FAISS index
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("supplychain_faiss_index")
    print("Vector DB created successfully!")

# Call the function
build_vectordb()