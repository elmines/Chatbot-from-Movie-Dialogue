import re
import nltk
import gensim
import pycontractions
import language_check

import os
import sys

import numpy as np #For shuffling


np.random.seed(1)

_INFINITY = float("inf")

_DEFAULT_MAX_WORDS = 12000
_DEFAULT_MIN_LINE_LENGTH = 1
_DEFAULT_MAX_LINE_LENGTH = 60


_DEFAULT_UNK = "<UNK>"
_DEFAULT_CONTR_MODEL = "w2vec_models/contractions.model"

class SimpleContractions(pycontractions.Contractions):

	def __init__(self, w2v_path):
		pycontractions.Contractions.__init__(self, w2v_path)
		self.w2v_model = gensim.models.KeyedVectors.load(self.w2v_path)
		self.lc_tool = language_check.LanguageTool(self.lang_code)



def gen_datasets(lines_path, conversations_path,
		max_vocab=_DEFAULT_MAX_WORDS, min_line_length=_DEFAULT_MIN_LINE_LENGTH, max_line_length=_DEFAULT_MAX_LINE_LENGTH,
		unk = _DEFAULT_UNK, contraction_model_path=None, partition=(0.8, 0.2, 0.0), output_dir="corpora", verbose=True):
	"""
	lines_path - path to the Cornell Movie lines file
	conversations_path - path to the Cornell Movie conversations path
	max_vocab - The maximum size vocabulary to generate (not counting the unknown token), default of 12000
	min_line_length - The minimum number of tokens for a prompt or answer, default of 1
	max_line_length - The maximum number of tokens for a prompt or answer, defalut of infinity
	unk - The symbol to be used for the unknown token, default of <UNK>
	contraction_model_path - The path to a gensim.models.KeyedVector file already trained on the text
	partition - 3-tuple of the ratios of the total dataset to use for training, validation, and testing
	output_dir - path-like object to which the cleaned text files and the vocabulary are written
	verbose - Print messages indicating the files that have been generated

	Note that the vocabulary file written to output_dir will always have the unknown token as its first entry
	"""

	
	if len(partition) != 3:
		raise ValueError("partition has length {}, must be 3.".format(len(partition)))
	if abs(sum(partition) - 1.0) > 0.001:
		raise ValueError("Ratios in partition must sum to 1.0")
	
	

	with open(lines_path, "r", encoding="utf-8", errors="ignore") as r:
		lines = r.read().split("\n")
	with open(conversations_path, "r", encoding="utf-8", errors="ignore") as r:
		conv_lines = r.read().split("\n")
	if verbose: sys.stderr.write("Read sequences from Cornell files\n")

	# Create a dictionary to map each line's id with its text
	ids = []
	lines_text = []
	for line in lines:
		_line = line.split(' +++$+++ ')
		if len(_line) == 5: #Lines not of length 5 are not properly formatted
			ids.append(_line[0])
			lines_text.append(_line[4])


	if contraction_model_path is None:
		orig_prompts = prompts
		orig_answers = answers
		corpus_tokens = [[token for token in prompt.split(" ")] for prompt in orig_prompts] + [[token for token in answer.split(" ")] for answer in orig_answers]	
		contraction_model_path = _DEFAULT_CONTR_MODEL
		contraction_model = gensim.models.Word2Vec(sentences=corpus_tokens, size=1024, window=5, min_count=1, workers=4, sg=0)
		model_vectors = contraction_model.wv
		model_vectors.save(contraction_model_path)
		if verbose: sys.stderr.write("Wrote Word2Vec model for finding contractions at {}.\n".format(contraction_model_path))
		#The model will have to be reloaded by pycontractions, so why waste memory? 
		del contraction_model
		del model_vectors
	contraction_exp = SimpleContractions(contraction_model_path)
	clean_text = _clean(lines_text, contraction_exp, verbose=verbose)
	if verbose: sys.stderr.write("Expanded contractions and both tokenized and lowercased the text.\n")
	del contraction_exp

	id2line = { id_no:line for (id_no, line) in zip(ids, clean_text) }
	(prompts, answers) = _generate_sequences(id2line, conv_lines)

	write_text("clean_prompts.txt", prompts)
	write_text("clean_answers.txt", answers)

	(short_prompts, short_answers) = _filter_by_length(prompts, answers, min_line_length, max_line_length)

	vocab = [unk] + _generate_vocab(short_prompts, short_answers, max_vocab)
	vocab2int = {word:index for (index, word) in enumerate(vocab) }
	if verbose: sys.stderr.write("Generated the vocabulary.\n")
	
	prompts_with_unk = _replace_unknowns(short_prompts, vocab2int, unk)
	answers_with_unk = _replace_unknowns(short_answers, vocab2int, unk)
	if verbose: sys.stderr.write("Replaced out-of-vocabulary words with {}.\n".format(unk))

	assert len(prompts_with_unk) == len(answers_with_unk)
	shuffled_indices = np.random.permutation(len(prompts_with_unk))
	shuffled_prompts = [prompts_with_unk[index] for index in shuffled_indices]
	shuffled_answers = [answers_with_unk[index] for index in shuffled_indices]
	if verbose: sys.stderr.write("Shuffled the dataset.\n")


	full_prompts = shuffled_prompts
	full_answers = shuffled_answers


	num_train = int(partition[0] * len(full_prompts))
	num_valid = int(partition[1] * len(full_prompts))
	num_test = len(full_prompts) - num_valid - num_train
	train_indices = (0, num_train)
	valid_indices = (num_train, num_train + num_valid)
	test_indices = (num_train + num_valid, -1)

	for (purpose, indices) in zip( ["train", "valid", "test"], [train_indices, valid_indices, test_indices] ):
		#Don't write a file if we didn't partition any data for that purpose
		if indices[1] > indices[0]:
			prompt_lines = full_prompts[indices[0]:indices[1]]
			prompts_path = os.path.join(output_dir, purpose + "_prompts.txt")
			write_text(prompts_path, prompt_lines)
			if verbose: sys.stderr.write("Wrote {} lines to {}.\n".format(indices[1] - indices[0], prompts_path))
	
			answer_lines = full_answers[indices[0]:indices[1]]
			answers_path = os.path.join(output_dir, purpose + "_answers.txt")
			write_text(answers_path, answer_lines)
			if verbose: sys.stderr.write("Wrote {} lines to {}.\n".format(indices[1] - indices[0], answers_path))
	vocab_path = os.path.join(output_dir, "vocab.txt")
	write_vocab(vocab_path, vocab2int)
	if verbose: sys.stderr.write("Wrote vocabulary to {}.\n".format(vocab_path))



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



def _generate_sequences(id2line, conv_lines):

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


def _clean(text, contractions_exp, verbose=False):
	"""
	"""
	if verbose: sys.stderr.write("{} sequences to clean.".format(len(text)))

	i = 0
	expanded_text = []
	for expansion in contractions_exp.expand_texts(text, precise=False):
		expanded_text.append(_punct_filters(expansion))
		i += 1
		if verbose and i % 1000 == 0:
			sys.stderr.write("Cleaned punctuation and contractions of {} sequences.\n".format(i))

	tokenized = [nltk.word_tokenize(sequence) for sequence in expanded_text]
	lowercased = _lowercase_tokens(tokenized)
	return lowercased
	
def _punct_filters(text):
	text = re.sub(r"\.+", ".", text)     #Ellipses and the like
	text = re.sub(r"\. \. \.", ".", text)
	text = re.sub(r"\-+", "-", text)
	text = re.sub(r"\?+", "?", text)     #Duplicate end punctuation
	text = re.sub(r"\!+", "!", text)

	return text

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

if __name__ == "__main__":
	gen_datasets("movie_lines.txt", "movie_conversations.txt", contraction_model_path = "w2vec_models/contractions.model")

