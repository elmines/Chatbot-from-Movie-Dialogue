import sys
print(sys.path)
import spacy

nlp = spacy.load("en_core_web_sm")
def lemmatize(token):
	lemma = str( nlp(token)[0].lemma_ )
	if lemma != "-PRON-":
		return lemma
	return None


def vocab_match(prop_vocab, targ_vocab, verbose=False):
	"""
	prop_vocab - Our \"proposed\" vocabulary of words we want to map to match_vocab
	targ_vocab - The vocab to which we're matching
	
	Returns
		A dictionary object mapping every word in prop_vocab to either a word in match_vocab or None
	"""
	
	list_neutral = []
	list_matched = []
	list_mapping = []
	matched_method = []

	identity = lambda word: word if word in targ_vocab else None

	lemma_list = [lemmatize(word) for word in targ_vocab]
	def by_lemma(word):
		lemma = lemmatize(word)
		if lemma is None: return None
		for i in range(len(lemma_list)):
			if lemma == lemma_list[i]:
				return targ_vocab[i]
		return None


	functions = []
	method_names = []
	functions.append(identity)
	functions.append(by_lemma)

	if verbose: sys.stderr.write("{} words to process for matching.\n".format(len(prop_vocab)))
	mapping = {}
	j = 1
	for word in prop_vocab:
		if verbose and j % 1000 == 0: sys.stderr.write("{} words processed for matching.\n".format(j))

		(matched, i) = (False, 0)
		while (not matched) and (i < len(functions)):
			target = functions[i](word)
			if target is not None:
				matched = True
				mapping[word] = target
			i += 1
		if not matched: mapping[word] = None
		j += 1
				
	return mapping
