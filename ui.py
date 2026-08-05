"""
ui.py
LASC Assistant icin Streamlit arayuzu. Is mantigi icermez -- sadece
answer_question() cagirir, tipki chat.py gibi.
"""
import streamlit as st

from src.llm import answer_question

st.title("LASC Assistant")
st.caption("Ask a question about the LASC documents.")

question = st.text_input("Your question:")
ask_clicked = st.button("Ask")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            result = answer_question(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.divider()

        st.subheader("Sources")
        for src in result["sources"]:
            st.write(f"- {src['file_name']} | Page {src['page']} | Chunk {src['chunk_index']}")