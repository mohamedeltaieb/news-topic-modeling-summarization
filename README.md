# News Topic Modeling and Summarization Project
<img width="1905" height="825" alt="image" src="https://github.com/user-attachments/assets/5a28ff50-7771-4b4f-be89-a7d187e9f52e" />

<img width="1668" height="906" alt="image" src="https://github.com/user-attachments/assets/88de20d1-cd2f-4ffb-8b29-54abda6b2a12" />

<img width="1856" height="922" alt="image" src="https://github.com/user-attachments/assets/fb16e8cb-0e5f-4d80-8fdc-45c6e4ed0476" />

## Overview

This project provides a comprehensive pipeline for analyzing news articles through topic modeling and summarization techniques. It processes raw news data, identifies latent topics using LDA (Latent Dirichlet Allocation), and generates both extractive and abstractive summaries of articles.

## Dataset

The project uses the **News Category Dataset** from Kaggle, which contains around 210,000 news headlines from HuffPost between 2012-2022. Each article includes:

- Category (42 news categories like POLITICS, WELLNESS, etc.)
- Headline
- Authors
- Link to article
- Short description
- Publication date

**Dataset Source**: [News Category Dataset on Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

## Features

### 1. Data Preprocessing
- Text cleaning (lowercasing, punctuation removal)
- Lemmatization and stopword removal using spaCy
- Combines headlines and short descriptions for analysis

### 2. Topic Modeling
- Implements LDA (Latent Dirichlet Allocation) using Gensim
- Visualizes topics with pyLDAvis
- Identifies dominant topics for each article

### 3. Summarization
- **Extractive Summarization**: Uses TextRank algorithm to select key sentences
- **Abstractive Summarization**: Leverages transformers (Falconsai/text_summarization model) to generate new summary text

### 4. Streamlit Interface
- Interactive web app to explore topics and summaries
- Adjustable number of topics
- Choice between extractive/abstractive summaries

## File Structure

```
News_Article_Topic_Modeler_and_Summarizer_project/
│
├── app/
│   └── streamlit_app.py
├── src/
│   ├── summarizer.py
│   ├── topic_modeler.py
│   └── preprocessing.py
├── data/
│   └── News_Category_Dataset_v3.json
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd news-topic-modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Download the dataset from Kaggle and place it in `data/` directory

## Usage

### Jupyter Notebooks
Run notebooks in sequence:
1. `1_data_cleaning.ipynb`
2. `2_topic_modeling.ipynb`
3. `3_extractive_summarization.ipynb`
4. `4_abstractive_summarization.ipynb`

### Streamlit App
```bash
streamlit run streamlit_app.py
```

## Requirements

The project requires the following key dependencies:

- Python 3.7+
- pandas
- numpy
- scikit-learn
- gensim
- spacy
- transformers
- streamlit
- pyLDAvis
- matplotlib
- seaborn

See `requirements.txt` for the complete list with specific versions.

## Results

The system provides:
- Visualizations of topic distributions
- Dominant topic assignments for each article
- High-quality summaries (both extractive and abstractive)
- Interactive exploration of news topics

## Technical Details

### Topic Modeling
- Uses Latent Dirichlet Allocation (LDA) with Gensim
- Configurable number of topics
- Topic visualization with pyLDAvis
- Coherence score evaluation

### Summarization Approaches
- **Extractive**: TextRank algorithm for sentence ranking
- **Abstractive**: Transformer-based model for generating new text

### Preprocessing Pipeline
- Text normalization and cleaning
- Lemmatization with spaCy
- Custom stopword removal
- Feature extraction for modeling

## Future Work

- Add more sophisticated preprocessing (NER, custom stopwords)
- Implement hierarchical topic modeling
- Add sentiment analysis
- Improve abstractive summarization with larger models
- Deploy as a web service
- Add support for real-time news feeds
- Implement topic evolution over time analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this project in your research, please cite the original dataset:

```bibtex
@article{misra2022news,
  title={News Category Dataset},
  author={Misra, Rishabh},
  journal={arXiv preprint arXiv:2209.11429},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuffPost for the original news articles
- Kaggle for hosting the dataset
- The open-source community for the tools and libraries used

---

For questions or support, please open an issue in the repository.
