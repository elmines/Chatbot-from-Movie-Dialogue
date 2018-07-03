
import multiprocessing as mp

#System utilities
import os
import sys

import numpy as np #For shuffling data

#Local modules
import testset

from preprocessing import clean


np.random.seed(1)

INFINITY = float("inf")

_DEFAULT_MAX_WORDS = 12000
_DEFAULT_MIN_LINE_LENGTH = 1
_DEFAULT_MAX_LINE_LENGTH = INFINITY


_DEFAULT_UNK = "<UNK>"
_DEFAULT_CONTR_MODEL = "w2vec_models/contractions.model"

def gen_datasets(lines_path, conversations_path,
		max_vocab=_DEFAULT_MAX_WORDS, min_line_length=_DEFAULT_MIN_LINE_LENGTH, max_line_length=_DEFAULT_MAX_LINE_LENGTH, unk = _DEFAULT_UNK, contraction_model_path=None, partition=(0.8, 0.2), output_dir="corpora", num_processes=1, verbose=True):
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
	num_processes - number of processes to use for multithreaded code
	verbose - Print messages indicating the files that have been generated

	Note that the vocabulary file written to output_dir will always have the unknown token as its first entry
	"""

	if len(partition) != 2:
		raise ValueError("partition has length {}, must be 2.".format(len(partition)))
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
			lines_text.append( _line[4] )

	id2line = { id_no:line for (id_no, line) in zip(ids, lines_text) }
	(prompts, answers) = _generate_sequences(id2line, conv_lines)
	sys.stderr.write("{} dialog exchanges.\n".format(len(prompts)))

	#remaining_indices = list( set(range(len(prompts))) - set(testset.test_indices("./test_set_removed.xlsx")) )
	#remaining_indices.sort()
	#prompts = [prompts[index] for index in remaining_indices]
	#answers = [answers[index] for index in remaining_indices]
	#if verbose: sys.stderr.write("{} sequences remaining after filtering out test sequences.\n".format(len(prompts)))

	clean_prompts = [ clean.pre_clean_seq(prompt).split() for prompt in prompts]
	clean_answers = [ clean.pre_clean_seq(answer).split() for answer in answers]

	(prompts, answers) = (clean_prompts, clean_answers)


	#joined_text = [token_sequence for token_sequence in prompts+answers] #Concatenate text for cleaning
	#clean_text = _clean(joined_text, contraction_exp, num_processes=num_processes, verbose=verbose)
	#(prompts, answers) = (clean_text[:len(prompts)], clean_text[len(prompts):])    #Break text back apart
	#assert len(prompts) == len(answers)

	if verbose: sys.stderr.write("Filtering sequences by length . . .\n")
	(short_prompts, short_answers) = _filter_by_length(prompts, answers, min_line_length, max_line_length)
	if verbose:
		sys.stderr.write("Filtered out sequences with less than {} or more than {} tokens; {} exchanges remaining.\n".format(min_line_length, max_line_length, len(short_prompts)))

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

	train_indices = []
	valid_indices = []
	for i, answer in enumerate(full_answers):
		if unk in answer:
			valid_indices.append(i)
		else:
			train_indices.append(i)

	for (purpose, indices) in zip( ["train", "valid"], [train_indices, valid_indices] ):
		prompt_lines = [full_prompts[index] for index in indices]
		prompts_path = os.path.join(output_dir, purpose + "_prompts.txt")
		write_text(prompts_path, prompt_lines)
		if verbose: sys.stderr.write("Wrote {} lines to {}.\n".format(len(prompt_lines), prompts_path))

		answer_lines = [full_answers[index] for index in indices]
		answers_path = os.path.join(output_dir, purpose + "_answers.txt")
		write_text(answers_path, answer_lines)
		if verbose: sys.stderr.write("Wrote {} lines to {}.\n".format(len(answer_lines), answers_path))
		
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



def _clean_proc(text, contractions_exp, proc_id="0", verbose=False):
	"""
	text - list(str) of text sequences
	contractions_exp - pycontractions.Contractions object for expanding contractions
	verbose - Print progress messages to stderr (as all those regex's take a while)
	"""
	i = 0
	expanded_text = []
	for expansion in contractions_exp.expand_texts(text, precise=False):
		expanded_text.append(expansion)
		i += 1
		if verbose and i % 1000 == 0:
			sys.stderr.write("Process {}: Cleaned contractions of {}/{} sequences.\n"
					.format(proc_id, i, len(text)))

	filtered = [_gen_filter(expansion) for expansion in expanded_text]
	filtered = [_web_filter(sequence) for sequence in filtered]
	filtered = [_misc_filter(sequence) for sequence in filtered]
	tokenized = _tokenize(filtered)
	lowercased = _lowercase_tokens(tokenized)
	return lowercased

def _clean(text, contractions_exp, num_processes=1, verbose=False):
	if num_processes < 2:
		sys.stderr.write("Cleaning text with 1 process.\n")
		return _clean_proc(text, contractions_exp, verbose=verbose)

	partition_size = len(text) // num_processes
	remainder = len(text) % num_processes
	partitions = []
	partitions.append(text[:partition_size + remainder])

	start_index = len(partitions[0])
	for i in range(1, num_processes):
		partitions.append( text[start_index:start_index+partition_size] )
		start_index += partition_size
	assert sum(len(partition) for partition in partitions) == len(text)


	expanders = [contractions_exp]*num_processes
	proc_ids = [str(i) for i in range(num_processes)]
	verbosities = [verbose]*num_processes
	args = zip(partitions, expanders, proc_ids, verbosities)

	pool = mp.Pool(processes = num_processes)
	if verbose: sys.stderr.write("Cleaning text with {} processes.\n".format(num_processes))
	results = pool.starmap(_clean_proc, args)

	cleaned_text = []
	for result in results:
		cleaned_text.extend(result)
	return cleaned_text

def _gen_filter(text):
	text = re.sub(r'[\?\.\!\-]+(?=[\?\.\!\-])', '', text) #Duplicate end punctuation
	text = re.sub( '\s+', ' ', text ).strip()             #Replace special whitespace characters with simple spaces
	#text = re.sub(r"\. \. \.", ".", text) # I see ellipses in the test set, so I'm leaving them for now

	return text

def _misc_filter(text):
	if np.random.random() < 0.90: #Use shall 10% of the time
		text = re.sub("shall", "will", text)
	return text

def _web_filter(text):
	text = re.sub("&quot;", ' " ', text)         
	text = re.sub("&amp;", ' & ', text)          
	text = re.sub("(<.*?>)|(&.*?;)", "", text)  
	return text


def _tokenize(sequences):
	tok_sequences = []
	for sequence in sequences:
		text = " ".join(nltk.word_tokenize(sequence))
		text = re.sub("can not", "cannot", text) #The Penn Treebannk tokenizer splits cannot into "can" and "not"
		text = re.sub("(``)|('')", ' " ', text)    #It also replaces opening double quotes with `` and closing ones with ''
		tok_sequences.append(text.split())

	return tok_sequences


def _lowercase_tokens(tokens):
	"""
	tokens - list(list(str)) where tokens[i][j] is the jth token of the ith sequence or sentence
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
	gen_datasets("movie_lines.txt", "movie_conversations.txt", max_line_length = 60)

