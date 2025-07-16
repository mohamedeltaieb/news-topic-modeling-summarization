from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline

summarizer = pipeline("summarization", model="Falconsai/text_summarization")


def extractive_summary(text, sentence_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)



def abstractive_summary(text, max_ratio=0.5, max_limit=60):
    if not isinstance(text, str) or not text.strip():
        return "N/A"

    input_len = len(text.split())
    max_new_tokens = max(8, min(int(input_len * max_ratio), max_limit))

    summary = summarizer(text, max_new_tokens=max_new_tokens, do_sample=False)
    return summary[0]['summary_text']
