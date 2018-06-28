import gensim
import numpy as np
import pandas as pd

import sys
import os

import match


def appended_vad(model=None, vocab2int=None, exclude=None, regenerate=True, verbose=True):
	"""
	model - a gensim Word2Vec model, already loaded
	regenerate - Actually create the VAD model, rather than just return a path
	exclude - list of tokens in vocab2int to assign the neutral vector

	If regenerate is True, model and vocab2int must be specified

	Returns
		A path to an .npy file containing the word embeddings
	"""

	model_path = os.path.join("word_Vecs_VAD.npy")
	if not regenerate: return model_path

	if (model is None) or (vocab2int is None):
		raise ValueError("Must specify model parameter if generating a VAD-appended model")

	df_vad=pd.read_excel('Warriner, Kuperman, Brysbaert - 2013 BRM-ANEW expanded.xlsx')

	df_vad["Word"] = df_vad["Word"].apply(str)
	targ_vocab = list(df_vad["Word"])

	mapping = match.vocab_match(model.wv.index2word, targ_vocab, verbose=verbose)

	if exclude is not None:
		for word in exclude:
			mapping[word] = None

	list_wordvecs=[]
	for i,word in enumerate(model.wv.index2word):
    		list_wordvecs.append(word)
	
	word_vecs_vad = np.zeros((len(model.wv.vocab),1027))
	count_vad=0
	count_neutral=0

	for i,word in enumerate(model.wv.index2word):
		target_word = mapping[word]
		if target_word is not None:
			if verbose:
				sys.stdout.write("VAD Values Assigned: {} --> {}\n".format(word, target_word))
			count_vad += 1
			word_vecs_vad[vocab2int[word]][0:1024] = model[word]
			word_vecs_vad[vocab2int[word]][1024] = df_vad.loc[df_vad['Word'] == target_word, 'V.Mean.Sum'].iloc[0]
			word_vecs_vad[vocab2int[word]][1025] = df_vad.loc[df_vad['Word'] == target_word, 'A.Mean.Sum'].iloc[0]
			word_vecs_vad[vocab2int[word]][1026] = df_vad.loc[df_vad['Word'] == target_word, 'D.Mean.Sum'].iloc[0]


		else:
			if verbose:
				sys.stdout.write("Neutral Vector Assigned: {}\n".format(word))
			count_neutral += 1
			word_vecs_vad[vocab2int[word]][0:1024] = model[word]
			word_vecs_vad[vocab2int[word]][1024]   = 5
			word_vecs_vad[vocab2int[word]][1025]   = 1
			word_vecs_vad[vocab2int[word]][1026]   = 5

	if verbose:
		sys.stderr.write("{}/{} words assigned corresponsing VAD values.\n".format(count_vad, len(model.wv.vocab)))
		sys.stderr.write("{}/{} words assigned the neutral VAD vector.\n".format(count_neutral, len(model.wv.vocab)))
	
	np.save(model_path,word_vecs_vad)
	return model_path


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
