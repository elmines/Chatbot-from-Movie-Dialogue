# Emotional Chatbot
A chatbot for emotional dialog generation built in TensorFlow 1.9 and modeled after the approaches described in [Asghar et al.](https://arxiv.org/abs/1709.03968) and [Koshla et al.](https://arxiv.org/abs/1805.07966).

## Usage
Check out [examples](examples/) to see how to train a model and then perform inference with it after training. The project's two "main" scripts are [train.py](train.py) and [infer.py](infer.py).

## Dependencies
- Python 3
- [tensorflow](https://pypi.org/project/tensorflow/) 1.9
- [gensim](https://pypi.org/project/gensim/) (for training Word2Vec embeddings)
- [pandas](https://pypi.org/project/pandas/)
  - [xlrd](https://pypi.org/project/xlrd/) 
- [spacy](https://pypi.org/project/spacy/) (for preprocessing submodule)
  - [en\_core\_web\_sm](https://spacy.io/usage/models) language model
- [matplotlib](https://pypi.org/project/matplotlib/) (for analysis submodule)
- [sphinx](https://pypi.org/project/Sphinx/) 1.7.6 or higher (for generating documentation)
