
from gensim.models import Word2Vec #pycontractions requires a Word2Vec model
import pycontractions
import re
import nltk

import numpy as np #For shuffling

np.random.seed(1)

_INFINITY = float("inf")

_DEFAULT_MAX_WORDS = _INFINITY
_DEFAULT_MIN_LINE_LENGTH = 1
_DEFAULT_MAX_LINE_LENGTH = _INFINITY


_DEFAULT_UNK = "<UNK>"
_DEFAULT_CONTR_MODEL = "contractions.model"


def gen_datasets(self, lines_path, conversations_path
		max_vocab=_DEFAULT_MAX_WORDS, min_line_length=_DEFAULT_MIN_LINE_LENGTH, max_line_length=_DEFAULT_MAX_LINE_LENGTH,
		unk = _DEFAULT_UNK, contraction_model_path = _DEFAULT_CONTR_MODEL):

	
	with open(lines_path, "r", encoding="utf-8", errors="ignore") as r:
			lines = r.read().split("\n")
	with open(conversations_path, "r", encoding="utf-8", errors="ignore") as r:
		conv_lines = r.read().split("\n")


	(prompts, answers) = _generate_sequences(lines, conv_lines)

	orig_prompts = prompts
	orig_answers = answers

	corpus_tokens = [[token for token in prompt.split(" ")] for token in prompt] +	
			[[token for token in answer.split(" ")] for token in answer]	
	contraction_model = Word2Vec(sentences=corpus_tokens, size=1024, window=5, min_count=1, workers=4, sg=0)
	contraction_model.save(contraction_model_path)
	del contraction_model #It will have to be reloaded by pycontractions, so why waste memory?


	(clean_prompts, clean_answers) = _clean(prompts, answers)



	(short_prompts, short_answers) = _filter_by_length(clean_prompts, clean_answers, min_line_length, max_line_length)

	#self.unk = unk
	vocab = _generate_vocab(short_prompts, short_answers, max_vocab) + [unk]
	vocab2int = { word:index for (index, word) in enumerate(self.vocab) }
	int2vocab = { index:word for (word, index) in self.vocab2int.items() }
	
	prompts_with_unk = Corpus._replace_unknowns(short_prompts, vocab2int, unk)
	answers_with_unk = Corpus._replace_unknowns(short_answers, vocab2int, unk)

	prompts_text = prompts_with_unk
	answers_text = answers_with_unk

	prompts_int = _encode(prompts, vocab2int)
	answers_int = _encode(answers, vocab2int)


	

def _generate_sequences(lines, conv_lines):
	# Create a dictionary to map each line's id with its text
	id2line = {}
	for line in lines:
		_line = line.split(' +++$+++ ')
		if len(_line) == 5: #Lines not of length 5 are not properly formatted
	        	id2line[_line[0]] = _line[4]

	# Create a list of all of the conversations' lines' ids.
	convs = []
	for line in conv_lines[:-1]:
	    	_line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
	    	convs.append(_line.split(','))
	# Sort the sentences into prompts (inputs) and answers (targets)
	prompts = []
	answers = []
	for conv in convs:
		for i in range(len(conv)-1):
			prompts.append(id2line[conv[i]])
			answers.append(id2line[conv[i+1]])
	return (prompts, answers)


def _clean(prompts, answers, contractions_model_path):
	"""
	Returns
		prompts - a 2-D list of strings where prompt[i][j] is the jth token in the ith sequence
		answers - a 2-D list of strings like prompts
	"""
	prompts_string = "\n".join(prompts)
	answers_string = "\n".join(answers)

	#Expand contractions
	expander = pycontractions.Contractions(contractions_model_path)
	#Each loop will only run once
	for expansion in expander.expand_texts(prompts_string, precise=True):
		expanded_prompts = expansion
	for expansion in expander.expand_texts(answers_string, precise=True):
		expanded_answers = expansion
	del expander #Get rid of that model

	prompts_tokens = [" ".split(prompt) for prompt in expanded_prompts.split("\n")]
	answers_tokens = [" ".split(answer) for answer in expanded_answers.split("\n")]

	prompts_tokenized = [nltk.word_tokenize(prompt) for prompt in prompts_tokens]
	answers_tokenized = [nltk.word_tokenize(answer) for answer in answers_tokens]

	lowercase_prompts = _lowercase_tokens(prompts_tokenized)
	lowercase_answers = _lowercase_tokens(answers_tokenized)
	


def _lowercase_tokens(tokens):
	"""
	tokens - A 2-D list of strings where tokens[i][j] is the jth token of the ith sequence or sentence
	"""
	return [ [word.lower() for word in tokenized if re.match("^[a-zA-Z]+", word)] ]
	
def _filter_by_length(prompts, answers, min_line_length, max_line_length):
	"""
	Returns
		short_prompts - a 2-D list of strings where short_prompts[i][j] is the jth token of the ith sequence
		short_answers - a 2-D list of strings like short_prompts
	"""
	# Filter out the prompts that are too short/long
	short_prompts_temp = []
	short_answers_temp = []
	for (i, prompt) in enumerate(prompts):
		if len(prompt) >= min_line_length and len(prompt) <= max_line_length:
			short_prompts_temp.append(prompt)
			short_answers_temp.append(answers[i])
	# Filter out the answers that are too short/long
	short_prompts = []
	short_answers = []
	for (i, answer) in enumerate(short_answers_temp):
		if len(answer) >= min_line_length and len(answer) <= max_line_length:
	        	short_answers.append(answer)
	        	short_prompts.append(short_prompts_temp[i])
	return (short_prompts, short_answers)
def _generate_vocab(prompts, answers, max_vocab):
	"""
	prompts - A 2-D list of strings
	answers - A 2-D list of strings
	"""
	word_freq = {}
	for prompt in prompts:
    		for word in prompt:
        		if word not in word_freq: word_freq[word]  = 1
        		else:                     word_freq[word] += 1
	for answer in answers:
    		for word in answer:
        		if word not in word_freq: word_freq[word]  = 1
        		else:                     word_freq[word] += 1


	sorted_by_freq = sorted(word_freq.keys(), key=lambda word: word_freq[word], reverse=True)
	del word_freq

		if len(sorted_by_freq) < max_vocab:
		vocab = sorted_by_freq
	else:
		vocab = sorted_by_freq[:max_vocab]
	return vocab
def _replace_unknowns(sequences, vocab, unk):
	"""
	sequences - A 2-D list of strings
	Returns
		A 2-D list of strings
	"""
	return [ [word if word in vocab else unk for word in sequence] for sequence in sequences ]
def _encode(sequences, vocab2int):
	"""
	sequences - 2-D list of strings
	Returns
		A 2-D list of integers
	"""	
	# Convert the text to integers. 
	return [ [vocab2int[word] for word in sequence] for sequence in sequences]
