import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def preprocess_articles(path):
    df = pd.read_json(path, lines=True)
    df = df.dropna(subset=['headline', 'short_description'])
    df['text'] = (df['headline'] + ' ' + df['short_description']).apply(clean_text).apply(lemmatize)
    return df[['category', 'text', 'headline', 'short_description']]
