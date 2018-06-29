import gensim
import numpy as np
import pandas as pd

import sys
import os

#Local modules
import match

def w2vec(model_path, text, vocab2int, embedding_size=1024, verbose=True):
	if verbose: sys.stderr.write("Learning Word2Vec embeddings on {} sequences . . .\n".format(len(text)))
	model = gensim.models.Word2Vec(sentences=text, size=embedding_size, window=5, min_count=1, workers=4, sg=0)
	word_vecs = np.zeros((len(model.wv.vocab),embedding_size))
	for i,word in enumerate(model.wv.index2word):
        	word_vecs[vocab2int[word]] = model[word]

	np.save(model_path,word_vecs)
	if verbose: sys.stderr.write("Wrote Word2Vec model to {}\n".format(model_path))

	return word_vecs


def appended_vad(model_path, embeddings, vocab2int, exclude=None, verbose=True):
	"""
	model - a gensim Word2Vec model, already loaded
	regenerate - Actually create the VAD model, rather than just return a path
	exclude - list of tokens in vocab2int to assign the neutral vector

	If regenerate is True, model and vocab2int must be specified

	Returns
		A path to an .npy file containing the word embeddings
	"""
	
	#Simple list of vocabulary items at their proper indices
	int2vocab = sorted(vocab2int.keys(), key=vocab2int.__getitem__)

	df_vad=pd.read_excel('Warriner, Kuperman, Brysbaert - 2013 BRM-ANEW expanded.xlsx')
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

	np.save(model_path,word_vecs_vad)

	if verbose:
		sys.stderr.write("{}/{} words assigned corresponsing VAD values.\n".format(count_vad, len(vocab2int)))
		sys.stderr.write("{}/{} words assigned the neutral VAD vector.\n".format(count_neutral, len(vocab2int)))
	
	return word_vecs_vad


def counterfitted(model=None, vocab2int=None, regenerate=True):
	"""
	model - a gensim Word2Vec model, already loaded
	regenerate - Actually create the counterfitted model, rather than just return a path

	If regenerate is True, model must be specified

	Returns
		A path to an .npy file containing the word embeddings
	"""

	model_path = os.path.join('word_Vecs_counterfit_affect.npy')
	if not regenerate: return model_path

	if (model is None) or (vocab2int is None):
		raise ValueError("Must specify model parameter if generating a VAD-appended model")

	# Load Google's pre-trained Word2Vec model.
	model_counterfit_affect = gensim.models.KeyedVectors.load_word2vec_format('./w2v_counterfit_append_affect.bin', binary=True)
	list_counterfit =list(model_counterfit_affect.wv.vocab.keys())
	dict_lemma_counterfit={}
	for word in list_counterfit:
    		dict_lemma_counterfit[lemmatize_text(word)]=word
	word_vecs_counterfit_affect = np.zeros((len(model.wv.vocab),303))
	list_word_not_found =[]
	for i,word in enumerate(model.wv.index2word):
    		lemma = lemmatize_text(word)
    		if lemma in dict_lemma_counterfit.keys():
        		word_vecs_counterfit_affect[vocab2int[word]] = model_counterfit_affect[dict_lemma_counterfit[lemma]]
    		else:
        		list_word_not_found.append(vocab2int[word])
	word_unknown = np.mean(word_vecs_counterfit_affect, axis=0)
	for i in list_word_not_found:
    		word_vecs_counterfit_affect[i] = word_unknown

	np.save(model_path, word_vecs_counterfit_affect)
	return model_path

def retrofitted(model=None, vocab2int=None, regenerate=True):
	"""
	model - a gensim Word2Vec model, already loaded
	regenerate - Actually create the retrofitted model, rather than just return a path

	If regenerate is True, model must be specified

	Returns
		A path to an .npy file containing the word embeddings
	"""

	model_path = os.path.join('word_Vecs_retrofit_affect.npy')
	if not regenerate: return model_path

	if (model is None) or (vocab2int is None):
		raise ValueError("Must specify model parameter if generating a VAD-appended model")

	model_retrofit_affect = gensim.models.KeyedVectors.load_word2vec_format('./w2v_retrofit_append_affect.bin', binary=True)


	list_retrofit =list(model_retrofit_affect.wv.vocab.keys())

	dict_lemma_retrofit={}
	for word in list_retrofit:
    		dict_lemma_retrofit[lemmatize_text(word)]=word

	word_vecs_retrofit_affect = np.zeros((len(model.wv.vocab),303))
	list_word_not_found_retro =[]
	for i,word in enumerate(model.wv.index2word):
    		lemma = lemmatize_text(word)
    		if lemma in dict_lemma_retrofit.keys():
        		word_vecs_retrofit_affect[vocab2int[word]] = model_retrofit_affect[dict_lemma_retrofit[lemma]]
    		else:
        		list_word_not_found_retro.append(vocab2int[word])

	word_unknown_retro = np.mean(word_vecs_retrofit_affect, axis=0)

	for i in list_word_not_found_retro:
    		word_vecs_retrofit_affect[i] = word_unknown_retro

	np.save(model_path, word_vecs_retrofit_affect)
	return model_path
