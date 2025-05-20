import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------Load environment variables-----------------#
load_dotenv()
GOOGLE_API_KEY = os.getenv("AIzaSyAAhSrn9L4NQDo38u_zwO8z_L6qcvwZiXM")

# --------------------Streamlit setup----------------------#
st.set_page_config(page_title="Supply Chain RAG Agents", layout="wide")
st.title("Automated Supply Chain Optimization using GenAI")

# ----------------Load FAISS Vector Store----------------- #
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("supplychain_faiss_index", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 4})

retriever = load_vector_db() if os.path.exists("supplychain_faiss_index") else None

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

Question: {{question}}
"""
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
agent_choice = st.selectbox("Select Agent", ["Forecasting", "Data Collection", "Optimization", "Alert"])
query = st.text_input("Ask your supply chain question")

# -----------------------Run selected agent----------------------#
if query and retriever:
    with st.spinner(f"Running {agent_choice} Agent with RAG..."):
        result = multi_agent_coordinator(query, agent_choice)

        st.subheader("Best Answer")
        st.write(result["result"])

        st.subheader("Source Chunks")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i + 1}**")
            st.code(doc.page_content[:500], language="markdown")

elif query:
    st.warning("Vector DB is not available. Please ensure 'supplychain_faiss_index' exists.")
