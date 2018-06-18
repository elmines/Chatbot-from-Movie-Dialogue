import re
import nltk

import gensim
import pycontractions
import language_check

import os

import numpy as np #For shuffling

np.random.seed(1)

_INFINITY = float("inf")

_DEFAULT_MAX_WORDS = _INFINITY
_DEFAULT_MIN_LINE_LENGTH = 1
_DEFAULT_MAX_LINE_LENGTH = _INFINITY


_DEFAULT_UNK = "<UNK>"
_DEFAULT_CONTR_MODEL = "w2vec_models/contractions.model"

class SimpleContractions(pycontractions.Contractions):

	def __init__(self, w2v_path):
		pycontractions.Contractions.__init__(self, w2v_path)
		self.w2v_model = gensim.models.KeyedVectors.load(self.w2v_path)
		self.lc_tool = language_check.LanguageTool(self.lang_code)



def gen_datasets(lines_path, conversations_path,
		max_vocab=_DEFAULT_MAX_WORDS, min_line_length=_DEFAULT_MIN_LINE_LENGTH, max_line_length=_DEFAULT_MAX_LINE_LENGTH,
		unk = _DEFAULT_UNK, contraction_model_path=None, partition=(0.8, 0.2, 0.0), corpus_dir="corpora", verbose=True):
	"""
	lines_path - path to the Cornell Movie lines file
	conversations_path - path to the Cornell Movie conversations path
	max_vocab - The maximum size vocabulary to generate (not counting the unknown token), default of {}
	min_line_length - The minimum number of tokens for a prompt or answer, default of {}
	max_line_length - The maximum number of tokens for a prompt or answer, defalut of infinity
	unk - The symbol to be used for the unknown token, default of {}
	contraction_model_path - The path to a gensim.models.KeyedVector file already trained on the text
	partition - 3-tuple of the ratios of the total dataset to use for training, validation, and testing
	corpus_dir - path-like object to which the cleaned text files and the vocabulary are written
	verbose - Print messages indicating the files that have been generated
	""".format(_DEFAULT_MAX_WORDS, _DEFAULT_MIN_LINE_LENGTH, _DEFAULT_UNK)

	
	if len(partition) != 3:
		raise ValueError("partition has length {}, must be 3.".format(len(partition)))
	if abs(sum(partition) - 1.0) > 0.001:
		raise ValueError("Ratios in partition must sum to 1.0")
	
	

	with open(lines_path, "r", encoding="utf-8", errors="ignore") as r:
		lines = r.read().split("\n")
	with open(conversations_path, "r", encoding="utf-8", errors="ignore") as r:
		conv_lines = r.read().split("\n")
	(prompts, answers) = _generate_sequences(lines, conv_lines)
	if verbose: print("Read sequences from Cornell files")

	if contraction_model_path is None:
		orig_prompts = prompts
		orig_answers = answers
		corpus_tokens = [[token for token in prompt.split(" ")] for prompt in orig_prompts] + [[token for token in answer.split(" ")] for answer in orig_answers]	
		contraction_model_path = _DEFAULT_CONTR_MODEL
		contraction_model = gensim.models.Word2Vec(sentences=corpus_tokens, size=1024, window=5, min_count=1, workers=4, sg=0)
		model_vectors = contraction_model.wv
		model_vectors.save(contraction_model_path)
		if verbose: print("Wrote Word2Vec model for finding contractions at {}".format(contraction_model_path))
		#The model will have to be reloaded by pycontractions, so why waste memory? 
		del contraction_model
		del model_vectors


	contraction_exp = SimpleContractions(contraction_model_path)
	(clean_prompts, clean_answers) = _clean(prompts, answers, contraction_exp)
	del contraction_exp
	if verbose: print("Expanded contractions and both tokenized and lowercased the text")

	(short_prompts, short_answers) = _filter_by_length(clean_prompts, clean_answers, min_line_length, max_line_length)

	vocab = _generate_vocab(short_prompts, short_answers, max_vocab) + [unk]
	vocab2int = {word:index for (index, word) in enumerate(vocab) }
	if verbose: print("Generated the vocabulary")
	
	prompts_with_unk = _replace_unknowns(short_prompts, vocab2int, unk)
	answers_with_unk = _replace_unknowns(short_answers, vocab2int, unk)
	if verbose: print("Replaced out-of-vocabulary words with {}".format(unk))

	assert len(prompts_with_unk) == len(answers_with_unk)

	shuffled_indices = np.random.permutation(len(prompts_with_unk))
	shuffled_prompts = [prompts_with_unk[i] for i in range(len(shuffled_indices))]
	shuffled_answers = [answers_with_unk[i] for i in range(len(shuffled_indices))]
	if verbose: print("Shuffled dataset")


	full_prompts = shuffled_prompts
	full_answers = shuffled_answers

	num_train = int(partition[0] * len(full_prompts))
	num_valid = int(partition[1] * len(full_prompts))
	num_test = len(full_prompts) - num_valid - num_train

	train_indices = (0, num_train)
	valid_indices = (num_train, num_train + num_valid)
	test_indices = (num_train + num_valid, -1)

	output_dir = "corpora"
	output_files = [""]

	for (purpose, indices) in zip( ["train", "valid", "test"], [train_indices, valid_indices, test_indices] ):
		#Don't write a file if we didn't partition any data for that purpose
		if indices[1] > indices[0]:
			prompt_lines = full_prompts[indices[0]:indices[1]]
			prompts_path = os.path.join(corpus_dir, purpose + "_prompts.txt")
			write_text(prompts_path, prompt_lines)
			if verbose: print("Wrote {} lines to {}".format(indices[1] - indices[0], prompts_path))
	
			answer_lines = full_answers[indices[0]:indices[1]]
			answers_path = os.path.join(corpus_dir, purpose + "_answers.txt")
			write_text(answers_path, answer_lines)
			if verbose: print("Wrote {} lines to {}".format(indices[1] - indices[0], answers_path))
	

	write_vocab("vocab.txt", vocab2int)


def write_text(path, text):
	"""
	text - a 2-D list of strings
	"""
	lines = "\n".join([" ".join(sequence) for sequence in text])
	write_lines(path, lines)
def write_vocab(path, vocab2int):
	lines = "\n".join( ["{0} {1}".format(word, index) for (word, index) in vocab2int.items()] )
	write_lines(path, lines)
def write_lines(path, lines):
	with open(path, "w", encoding="utf-8") as w:
		w.write(lines)		


	

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


def _clean(prompts, answers, contractions_exp):
	"""
	prompts - a list of strings
	answers - a list of strings
	contractions_exp - an instance of pycontractions.Contractions
	Returns
		prompts - a 2-D list of strings where prompt[i][j] is the jth token in the ith sequence
		answers - a 2-D list of strings like prompts
	"""


	expanded_prompts = [expansion for expansion in contractions_exp.expand_texts(prompts, precise=True)]
	expanded_answers = [expansion for expansion in contractions_exp.expand_texts(answers, precise=True)]

	prompts_tokenized = [nltk.word_tokenize(prompt) for prompt in expanded_prompts]
	answers_tokenized = [nltk.word_tokenize(answer) for answer in expanded_answers]

	lowercase_prompts = _lowercase_tokens(prompts_tokenized)
	lowercase_answers = _lowercase_tokens(answers_tokenized)

	return (lowercase_prompts, lowercase_answers)
	


def _lowercase_tokens(tokens):
	"""
	tokens - A 2-D list of strings where tokens[i][j] is the jth token of the ith sequence or sentence
	"""
	return [[token.lower() if re.match("^[a-zA-Z]+", token) else token for token in sequence] for sequence in tokens]
	
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

