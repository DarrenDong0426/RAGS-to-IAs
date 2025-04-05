import streamlit as st
from rag_pipeline import chunk_and_retrieve, retrieve_info, generate_rag_response, model

# Run once to build vector DB and embeddings
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge():
    return chunk_and_retrieve()

embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed = load_knowledge()

# App title
st.title("EECS 487 Chatbot")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask something about the course..."):
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Retrieve relevant docs and generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_docs = retrieve_info(user_query, embedding_model, KNOWLEDGE_VECTOR_DATABASE, docs_processed)
            response = generate_rag_response(user_query, retrieved_docs, model)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
