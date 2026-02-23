import streamlit as st
from scripts.rag_core import answer_query

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Ryan Shen | Private RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("🤖RAG Interview & Project Assistant🤖")

st.markdown(
    """
Hi, my name is **Yiming (Ryan) Shen**.  
I am currently pursuing a **Master’s degree in Data Science and Artificial Intelligence** at the **University of Waterloo**.

Feel free to ask questions about my background, technical projects, or work experience.
"""
)

st.divider()

# ---------------------------------------------------
# Password Gate
# ---------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.subheader("🔐 Private Access")

    password = st.text_input("Enter password", type="password")

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.success("Access granted ✔")
            st.rerun()   # 🔥 Important: refresh app
        else:
            st.error("Incorrect password.")

    st.stop()

# ---------------------------------------------------
# Logout Button (Sidebar)
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### Session Control")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ---------------------------------------------------
# Main Query UI
# ---------------------------------------------------
st.subheader("Ask a Question")

query = st.text_input("Enter your question:")

col1, col2 = st.columns([1, 3])

with col1:
    generate = st.button("Generate Answer")

if generate:

    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):

            answer = answer_query(query)

        st.divider()
        st.markdown("### Answer")
        st.write(answer)

        st.divider()
        st.markdown(
            """
If you'd like to connect, you can reach me at:

- **y336shen@uwaterloo.ca** (University email)  
- **yimingshen20000719@gmail.com** (Personal email)
"""
        )
