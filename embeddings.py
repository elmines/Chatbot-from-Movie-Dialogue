"""
Utility module/script for generating word embeddings
"""
#System utilities
import sys
import os
import argparse
from copy import deepcopy

#Computation
import gensim
import numpy as np
import pandas as pd

#Local modules
import match

def create_parser():
	parser = argparse.ArgumentParser(description="Generate word embeddings given a corpus and vocabulary")

	parser.add_argument("--vocab", "-v", required=True, metavar="vocab.txt", help="Vocabulary file where each line is"
                                                                    " a word and its integer index, like so: \"jelly 272\"")
	parser.add_argument("--save-path", required=True, metavar="new_embeddings.npy", help="Path at which to save the newly generated embeddings as a .npy file")

	parser.add_argument("--size", default=1024, type=int, metavar="N", help="Embedding size for Word2Vec embeddings")
	parser.add_argument("--w2v", metavar="corpus.txt", help="Generate Word2Vec embeddings from the corpus")

	parser.add_argument("--retrofit", nargs=2, metavar=("embeddings.npy", "lexicon.txt"), help="Retrofit existing word embeddings stored in a Numpy file using Faruqui et al.-style lexicon")
	parser.add_argument("--counterfit", metavar="embeddings.npy", help="Counterfit existing word embeddings stored in a .npy or .npz file")

	return parser

_VAD_spreadsheet = os.path.join("resources", "Warriner, Kuperman, Brysbaert - 2013 BRM-ANEW expanded.xlsx")
if not os.path.exists(_VAD_spreadsheet):
	raise OSError("{} was not found and is a required resource for generating affective embeddings.".format(_VAD_spreadsheet))

def w2vec(text, vocab2int, embedding_size=1024, verbose=True):
	"""
	Generates Word2Vec embeddings

	:param list(list(str))          text: Tokenized sentences from which to generate the embeddings
	:param dict(str,int)       vocab2int: A mapping from tokens in the vocabulary to their integer indices in the range [0, vocab_size)
	:param int            embedding_size: The number of dimensions for each word embedding
	:param bool                  verbose: Print helpful messages to stderr
	
	:returns: The generated embeddings
	:rtype:   np.ndarray
	"""
	if verbose: sys.stderr.write("Learning Word2Vec embeddings on {} sequences . . .\n".format(len(text)))
	model = gensim.models.Word2Vec(sentences=text, size=embedding_size, window=5, min_count=1, workers=4, sg=0)

	word_vecs = np.zeros( (len(vocab2int),embedding_size) )
	for (word, index) in vocab2int.items():
		word_vecs[index] = model[word]
	return word_vecs

def appended_vad(embeddings, vocab2int, exclude=None, verbose=True):
	"""
	Appends VAD (Valence, Arousal, Dominance) values to existing word embeddings

	:param np.ndarray      embeddings: The original embeddings
	:param dict(str,int)    vocab2int: A mapping from tokens in the vocabulary to their integer indices
	:param list(str)          exclude: A list of tokens in vocab2int for which to assign the neutral vector (such as the unknown token)
	:param bool               verbose: Print helpful messages to stderr
	
	:returns: The generated embeddings
	:rtype:   np.ndarray
	"""
	#Simple list of vocabulary items at their proper indices
	int2vocab = sorted(vocab2int.keys(), key=vocab2int.__getitem__)

	df_vad=pd.read_excel(_VAD_spreadsheet)
	df_vad["Word"] = df_vad["Word"].apply(str)
	targ_vocab = list(df_vad["Word"])

	mapping = match.vocab_match(int2vocab, targ_vocab, verbose=verbose)
	if exclude is not None:
		for word in exclude:
			mapping[word] = None

	embedding_size = embeddings.shape[1]
	word_vecs_vad = np.zeros( (len(vocab2int), embedding_size+3) )
	count_vad=0
	count_neutral=0
	for word in int2vocab:
		target_word = mapping[word]
		index = vocab2int[word]
		if target_word is not None:
			if verbose: sys.stderr.write("VAD Values Assigned: {} --> {}\n".format(word, target_word))
			count_vad += 1
			word_vecs_vad[index][:-3] = embeddings[index]
			word_vecs_vad[index][-3] = df_vad.loc[df_vad['Word'] == target_word, 'V.Mean.Sum'].iloc[0]
			word_vecs_vad[index][-2] = df_vad.loc[df_vad['Word'] == target_word, 'A.Mean.Sum'].iloc[0]
			word_vecs_vad[index][-1] = df_vad.loc[df_vad['Word'] == target_word, 'D.Mean.Sum'].iloc[0]
		else:
			if verbose: sys.stderr.write("Neutral Vector Assigned: {}\n".format(word))
			count_neutral += 1
			word_vecs_vad[index][:-3] = embeddings[index]
			word_vecs_vad[index][-3]   = 5
			word_vecs_vad[index][-2]   = 1
			word_vecs_vad[index][-1]   = 5

	if verbose:
		sys.stderr.write("{}/{} words assigned corresponsing VAD values.\n".format(count_vad, len(vocab2int)))
		sys.stderr.write("{}/{} words assigned the neutral VAD vector.\n".format(count_neutral, len(vocab2int)))
	
	return word_vecs_vad


def aff2vec(model_path, vocab2int, aff_embeddings_path="./w2_counterfit_append_affect.bin", exclude=None, verbose=True):
	"""
	Extracts Aff2Vec embeddings for an arbitary vocabulary from an existing word embeddings .bin file

	For words in the vocabulary but not the .bin file, the function creates a \"neutral embedding\" which is simply the average of all the embedding
	vectors assigned for words that were found.

	:param path-like               model_path: The path at which to save the new embeddings before returning
	:param dict(str,int)            vocab2int: A mapping from tokens in the vocabulary to their integer indices
	:param path-like      aff_embeddings_path: Path to a .bin gensim file
	:param list(str)                  exclude: A list of tokens in vocab2int for which to assign the neutral vector (such as the unknown token)
	:param bool                       verbose: Print helpful messages to stderr
	
	:returns: The extracted embeddings
	:rtype:   np.ndarray
	"""

	#Simple list of vocabulary items at their proper indices
	#FIXME: Assumes the lowest index is indeed 0. Bad?
	int2vocab = sorted(vocab2int.keys(), key=vocab2int.__getitem__)

	# Load Google's pre-trained Word2Vec model.
	aff_embeddings = gensim.models.KeyedVectors.load_word2vec_format(aff_embeddings_path, binary=True)

	targ_vocab = list(aff_embeddings.wv.vocab.keys())
	mapping = match.vocab_match(int2vocab, targ_vocab, verbose=verbose)
	if exclude is not None:
		for word in exclude:
			mapping[word] = None

	embedding_size = len(aff_embeddings.wv[targ_vocab[0]])
	word_vecs_emot = np.zeros( (len(vocab2int), embedding_size) )

	assign_emot = []
	assign_neutral = []

	for word in int2vocab:
		target_word = mapping[word]
		word_index = vocab2int[word]
		if target_word is not None:
			if verbose: sys.stderr.write("Emotional Embeddings Assigned: {} --> {}\n".format(word, target_word))
			assign_emot.append(word_index)
			word_vecs_emot[word_index] = aff_embeddings.wv[target_word]

		else:
			if verbose: sys.stderr.write("Neutral Vector Assigned: {}\n".format(word))
			assign_neutral.append(word_index)


	neut_embedding = np.mean( word_vecs_emot[assign_emot], axis = 0)
	word_vecs_emot[assign_neutral] = neut_embedding
	np.save(model_path, word_vecs_emot)

	if verbose:
		sys.stderr.write("{}/{} words assigned emotional embeddings.\n".format(len(assign_emot), len(vocab2int)))
		sys.stderr.write("{}/{} words assigned the neutral vector.\n".format(len(assign_neutral), len(vocab2int)))

	return word_vecs_emot

def read_lexicon(filename):
	"""
	Read in a lexicon of the format used by Faruqui et al. at https://github.com/mfaruqui/retrofitting

	That is, each line consists of the keyword, followed by its corresponding words in the lexicon,
	all delimited by spaces. For example, the following line contains \"newspaper\" and two synonyms:

	newspaper journal tribune

	:param path-like filename: The lexicon file to be read

	:returns: A mapping from each keywords to its corresponding words in the lexicon
	:rtype: dict(str,list(str))
	""" 
	lexicon = {}
	with open(filename, "r", encoding="utf-8") as r:
		for line in r.readlines():
			words = line.lower().strip().split()
			lexicon[words[0]] = [word for word in words[1:]]
	return lexicon

def retrofit(embeddings, word2int, lexicon, numIters):
	"""
	Retrofit word vectors to a lexicon

	Modified from Faruqui et al.'s code at https://github.com/mfaruqui/retrofitting/retrofit.py

	:param np.ndarray          embeddings: The word embeddings to be retrofitted
	:param dict(str,int)         word2int: Mapping from a word to its index in embeddings

	:param dict(str,list(str))   lexicon: Mapping from a word to a list of corresponding words (be they synonyms, paraphrases, hyponyms, etc.)
	:param int                  numIters: The number of training iterations to perfrom

	:returns: The retrofitted vectors
	:rtype: dict(str,np.ndarray)
	"""

	#Added code
	#Mapping from a word to its embedding
	wordVecs = {word:embeddings[index] for (word, index) in word2int.items()}


	newWordVecs = deepcopy(wordVecs)
	wvVocab = set(newWordVecs.keys())
	loopVocab = wvVocab.intersection(set(lexicon.keys()))
	#print("size of loopVocab=", len(loopVocab))
	for it in range(numIters):
		#print("Iteration", it)
		# loop through every node also in ontology (else just use data estimate)
		for word in loopVocab:
			wordNeighbours = set(lexicon[word]).intersection(wvVocab)
			numNeighbours = len(wordNeighbours)
			#no neighbours, pass - use data estimate
			#print("\tneighbors of \"", word, "\":", wordNeighbours)
			if numNeighbours == 0:
				continue
			# the weight of the data estimate if the number of neighbours
			newVec = numNeighbours * wordVecs[word]
			# loop over neighbours and add to new vector (currently with weight 1)
			for ppWord in wordNeighbours:
				newVec += newWordVecs[ppWord]
			newWordVecs[word] = newVec/(2*numNeighbours)


	#Added code
	sorted_words = sorted(word2int, key=word2int.get)
	#print(sorted_words[:5])
	new_embeddings = np.stack( [newWordVecs[word] for word in sorted_words] )
	for word in word2int:
		assert np.all(newWordVecs[word] == new_embeddings[word2int[word]])
	#print("All assertions passed")

	return new_embeddings


if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()

	summation = sum(bool(option) for option in [args.w2v, args.retrofit, args.counterfit])
	if summation != 1:
		raise ValueError("You must choose exactly one of --w2v, --retrofit, or --counterfit")

	word2int = {}
	with open(args.vocab, "r", encoding="utf-8") as r:
		for line in r.readlines():
			[word, index] = line.split()
			word2int[word] = int(index)


	if args.w2v:
		with open(args.w2v, "r", encoding="utf-8") as r:
			tokens = [line.split() for line in r.readlines()]
		embeddings = w2vec(tokens, word2int, embedding_size=args.size) 
	elif args.retrofit:

		original_embeddings = np.load(args.retrofit[0])
		lexicon = read_lexicon(args.retrofit[1])
		embeddings = retrofit(original_embeddings, word2int, lexicon, numIters=10)

	np.save(args.save_path, embeddings)

