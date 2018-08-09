# Emotional Chatbot
A chatbot for emotional dialog generation built in TensorFlow 1.9 and modeled after the approaches described in [Asghar et al.](https://arxiv.org/abs/1709.03968) and [Koshla et al.](https://arxiv.org/abs/1805.07966).

## Usage
Check out [examples](examples/) to see how to train a model and then perform inference with it after training. The project's two "main" scripts are [train.py](train.py) and [infer.py](infer.py).

To generate some more expansive documentation, simply run [gen\_doc.sh](gen_doc.sh).

## Dependencies
- Python 3
- [tensorflow](https://pypi.org/project/tensorflow/) 1.9 or higher
- [gensim](https://pypi.org/project/gensim/) (for training Word2Vec embeddings)
- [PyYAML](https://pypi.org/project/PyYAML/) (for configuring a model)
- [numpy](https://pypi.org/project/numpy/)
- [sklearn](https://pypi.org/project/scikit-learn/) (for principal component analysis)
- [pandas](https://pypi.org/project/pandas/) (for spreadsheet processing)
  - [xlrd](https://pypi.org/project/xlrd/) 
- [spacy](https://pypi.org/project/spacy/) (for preprocessing package)
  - [en\_core\_web\_sm](https://spacy.io/usage/models) language model
- [matplotlib](https://pypi.org/project/matplotlib/) (for analysis package)
- [sphinx](https://pypi.org/project/Sphinx/) 1.7.6 or higher (for generating documentation)
