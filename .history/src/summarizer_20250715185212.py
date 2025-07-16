from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline

summarizer = pipeline("summarization")


def extractive_summary(text, sentence_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


abstractive_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text, max_ratio=0.5, max_limit=60):
    input_len = len(text.split())
    max_len = max(8, min(int(input_len * max_ratio), max_limit))
    return summarizer(text, max_length=max_len, min_length=5, do_sample=False)[0]['summary_text']