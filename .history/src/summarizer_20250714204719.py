from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline


def extractive_summary(text, sentence_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


abstractive_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text, max_length=60):
    summary = abstractive_pipeline(text, max_length=max_length, min_length=15, do_sample=False)
    return summary[0]['summary_text']