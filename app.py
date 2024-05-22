import pandas as pd
import pandasai as pdai
import streamlit as st
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe

st.title("Analyze Your Data")


model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="mistral"
)

films_csv = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

if films_csv is not None:
    films_df = pd.read_csv(films_csv)
    st.write(films_df.head())

    smart_df = SmartDataframe(films_df, config={"llm": model})
    prompt = st.text_area("Enter you query:")

    if st.button("Generate"):
        if prompt:
            with st.spinner(" Generating Response ..."):
                st.write(smart_df.chat(prompt))

