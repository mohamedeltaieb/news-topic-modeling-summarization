import os
import sys
import warnings

# Suppress TensorFlow and deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.preprocessing import preprocess_articles
from src.topic_modeler import build_lda_model, get_dominant_topic
from src.summarizer import extractive_summary, abstractive_summary

st.title("News Topic Modeler & Summarizer")

@st.cache_data
def load_data():
    return preprocess_articles("data/News_Category_Dataset_v3.json")

# Sidebar config
st.sidebar.header("Configuration")
n_topics = st.sidebar.slider("Number of Topics", 5, 20, 10)
summary_type = st.sidebar.selectbox("Summary Type", ["Extractive", "Abstractive"])

if st.button("Run Model"):
    with st.spinner("Loading and preprocessing data..."):
        df = load_data().sample(100).copy()

    with st.spinner("Building LDA topic model..."):
        lda_model, corpus, dictionary = build_lda_model(df['text'], num_topics=n_topics)
        df['topic'] = get_dominant_topic(lda_model, corpus)

    topic_choice = st.selectbox("Select Topic", sorted(df['topic'].unique()))
    topic_df = df[df['topic'] == topic_choice]

    st.write(f"Showing top 5 articles from Topic {topic_choice}:")

    with st.spinner("Generating summaries..."):
        for i, row in topic_df.head(5).iterrows():
            st.subheader(row['headline'])
            st.write(row['short_description'])
            if summary_type == "Extractive":
                summary = extractive_summary(row['text'])
            else:
                # Truncate text to 512 tokens
                truncated_text = " ".join(row['text'].split()[:512])
                summary = abstractive_summary(truncated_text)
            st.success(summary)
