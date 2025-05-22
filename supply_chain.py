import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# -----------------Load environment variables-----------------#
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_AAIzaSyAAhSrn9L4NQDo38u_zwO8z_L6qcvwZiXMPI_KEY")

# --------------------Streamlit setup----------------------#
st.set_page_config(page_title="Supply Chain RAG Agents", layout="wide")
st.title("Automated Supply Chain Optimization using GenAI")

# ----------------------- Upload and Process PDF ------------------------ #
uploaded_file = st.file_uploader("Upload a Supply Chain PDF document", type="pdf")

@st.cache_resource
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        loader = PyPDFLoader(tmp_file.name)
        documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

retriever = process_pdf(uploaded_file) if uploaded_file else None

# -----------------------Initialize Google Gemini LLM-----------------------#
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    google_api_key="AIzaSyAAhSrn9L4NQDo38u_zwO8z_L6qcvwZiXM"
)

# --------------------Core RAG logic --------------------- #
def run_rag_agent(query, agent_prompt):
    template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""You are an AI assistant.

{agent_prompt}

Use the following context to assist your answer:
{{context}}

Question: {{question}}"""
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": template},
        return_source_documents=True
    )
    return chain({"query": query})

# ----------------------- Agent coordination logic ------------------------#
def multi_agent_coordinator(query, agent_name):
    if agent_name == "Data Collection":
        agent_prompt = "You are a Data Collection Agent for a supply chain system. Analyze historical data, supplier lead times, and market trends."
    elif agent_name == "Forecasting":
        agent_prompt = "You are a Forecasting Agent. Predict future product demand based on historical trends and supply chain dynamics."
    elif agent_name == "Optimization":
        agent_prompt = "You are an Optimization Agent. Recommend strategies to reduce costs and optimize inventory levels."
    elif agent_name == "Alert":
        agent_prompt = "You are an Alert Agent. Detect and explain risks of stockouts or overstock and suggest mitigation actions."
    else:
        agent_prompt = "You are a general supply chain assistant."

    return run_rag_agent(query, agent_prompt)

# -----------------UI: Select agent and input query---------------------#
if uploaded_file:
    agent_choice = st.selectbox("Select Agent", ["Forecasting", "Data Collection", "Optimization", "Alert"])
    query = st.text_input("Ask your supply chain question")

    if query:
        with st.spinner(f"Running {agent_choice} Agent with RAG..."):
            result = multi_agent_coordinator(query, agent_choice)

            st.subheader("Best Answer")
            st.write(result["result"])

            st.subheader("Source Chunks")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i + 1}**")
                st.code(doc.page_content[:500], language="markdown")
else:
    st.info("Please upload a PDF to begin.")
