import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_topic_distribution(df):
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x='topic')
    plt.xticks(rotation=45)
    plt.title("Articles per Topic")
    plt.tight_layout()
    plt.show()


def generate_wordcloud(texts, topic_id):
    text = " ".join(texts)
    wc = WordCloud(width=800, height=400).generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for Topic {topic_id}")
    plt.show()
