from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


def build_lda_model(texts, num_topics=10):
    tokenized = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda_model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, corpus, dictionary

def visualize_topics(lda_model, corpus, dictionary, output_html="lda.html"):
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_html)


def get_dominant_topic(lda_model, corpus):
    return [sorted(lda_model[doc], key=lambda x: -x[1])[0][0] for doc in corpus]
