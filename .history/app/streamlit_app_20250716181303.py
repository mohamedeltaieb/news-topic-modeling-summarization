import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.preprocessing import preprocess_articles
from src.topic_modeler import build_lda_model, get_dominant_topic
from src.summarizer import extractive_summary, abstractive_summary


st.title("News Topic Modeler & Summarizer")

@st.cache_data

def load_data():
    df = preprocess_articles("data/News_Category_Dataset_v3.json")
    return df

st.sidebar.header("Configuration")
n_topics = st.sidebar.slider("Number of Topics", 5, 20, 10)
summary_type = st.sidebar.selectbox("Summary Type", ["Extractive", "Abstractive"])

if st.button("Run Model"):
    df = load_data()
    lda_model, corpus, dictionary = build_lda_model(df['text'], num_topics=n_topics)
    df['topic'] = get_dominant_topic(lda_model, corpus)

    topic_choice = st.selectbox("Select Topic", sorted(df['topic'].unique()))
    topic_df = df[df['topic'] == topic_choice]

    st.write(f"Showing top 5 articles from Topic {topic_choice}:")

    for i, row in topic_df.head(5).iterrows():
        st.subheader(row['headline'])
        st.write(row['short_description'])
        if summary_type == "Extractive":
            summary = extractive_summary(row['text'])
        else:
            summary = abstractive_summary(row['text'])
        st.success(summary)
