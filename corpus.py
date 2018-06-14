import re
import nltk

_INFINITY = float("inf")

_DEFAULT_MAX_WORDS = _INFINITY

_DEFAULT_MIN_LINE_LENGTH = 0
_DEFAULT_MAX_LINE_LENGTH = _INFINITY

_DEFAULT_EOS = "<EOS>"
_DEFAULT_PAD = "<PAD>"
_DEFAULT_UNK = "<UNK>"
_DEFAULT_GO = "<GO>"


class Corpus(object):
	"""
	A class for representing parallel corpora

	`self.unk` - The unknown word token
	`prompts` - A list of strings, each of which is a prompt from the corpus
	`answers` - A list of strings, where answers[i] is the reply to prompts[i]
	`prompts_int` - A list of integer lists, where each element is an integer that uniquely identifies a word
	`answers_int` - A list of integer lists, where each element is an integer that uniquely identifes a word
	`vocab` - A list of strings, including `metatokens`, each of which is a word in the vocabulary
	`vocab2int` - A dictionary mapping from a word (string) to an integer (index)
	"""


	def __init__(self, lines_path, conversations_path,
		max_vocab=_DEFAULT_MAX_WORDS, min_line_length=_DEFAULT_MIN_LINE_LENGTH, max_line_length=_DEFAULT_MAX_LINE_LENGTH,
		unk = _DEFAULT_UNK):

		"""
		`lines_path` - The path to the Cornell movie lines file
		`conversations_path` - The path to the Cornell movie conversations file
		`max_vocab` - The maximum vocabulary size
		`min_line_length` - The minimum length of a prompt or answer
		`max_line_length` - The maximum length of a prompt or answer
		`unk` - The unknown word token to be used (always the last element in the vocabulary)
		"""

	
		with open(lines_path, "r", encoding="utf-8", errors="ignore") as r:
			lines = r.read().split("\n")
		with open(conversations_path, "r", encoding="utf-8", errors="ignore") as r:
			conv_lines = r.read().split("\n")


		(prompts, answers) = Corpus._generate_sequences(lines, conv_lines)

		self.orig_prompts = prompts
		self.orig_answers = answers

		(clean_prompts, clean_answers) = Corpus._clean(prompts, answers)
		(short_prompts, short_answers) = Corpus._filter_by_length(clean_prompts, clean_answers, min_line_length, max_line_length)

		self.unk = unk
		self.vocab = Corpus._generate_vocab(short_prompts, short_answers, max_vocab) + [self.unk]
		self.vocab2int = { word:index for (index, word) in enumerate(self.vocab) }
		self.int2vocab = { index:word for (word, index) in self.vocab2int.items() }
	
		prompts_with_unk = Corpus._replace_unknowns(short_prompts, self.vocab2int, self.unk)
		answers_with_unk = Corpus._replace_unknowns(short_answers, self.vocab2int, self.unk)
		self.prompts = prompts_with_unk
		self.answers = answers_with_unk

		self.prompts_int = Corpus._encode(self.prompts, self.vocab2int)
		self.answers_int = Corpus._encode(self.answers, self.vocab2int)


	def write_prompts(self, path):
		"""
		Write the `prompts` text to a specified `path`
	
		`path` - The path to which to write the text
		"""
		Corpus._write_lines(path, self.prompts)
		
	def write_answers(self, path):
		"""
		Write the `answers` text to a specified `path`
	
		`path` - The path to which to write the text
		"""
		Corpus._write_lines(path, self.answers)

	def write_vocab(self, path):
		"""
		Write `vocabulary` to a file where each line is the word and its integer index separated by a space
		"""
		lines = [ "{} {}".format(word, index) for (word, index) in self.vocab2int.items() ]
		Corpus._write_lines(path, lines)
	
	

	@staticmethod
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


	@staticmethod
	def _clean(prompts, answers):
		"""
		Returns
			prompts - a 2-D list of strings where prompt[i][j] is the jth token in the ith sequence
			answers - a 2-D list of strings like prompts
		"""

		clean_prompts = []
		for prompt in prompts:
	    		clean_prompts.append(Corpus.clean_sequence(prompt))

		clean_answers = []    
		for answer in answers:
	    		clean_answers.append(Corpus.clean_sequence(answer))

		return (clean_prompts, clean_answers)

		
	@staticmethod
	def clean_sequence(sequence):
		"""
		Returns
			A list of strings where element i is the ith token
		"""
		filtered = Corpus._re_filters(sequence)
		tokenized = nltk.word_tokenize(filtered)
		lowercased = [word.lower() for word in tokenized if re.match('^[a-zA-Z]+', word) ]

		return lowercased

	@staticmethod
	def _re_filters(text):
		'''Clean text by removing unnecessary characters and altering the format of words.'''
		text = text.lower()
		text = re.sub(r"i'm", "i am", text)
		text = re.sub(r"he's", "he is", text)
		text = re.sub(r"she's", "she is", text)
		text = re.sub(r"it's", "it is", text)
		text = re.sub(r"that's", "that is", text)
		text = re.sub(r"what's", "what is", text)
		text = re.sub(r"where's", "where is", text)
		text = re.sub(r"how's", "how is", text)
		text = re.sub(r"\'ll", " will", text)
		text = re.sub(r"\'ve", " have", text)
		text = re.sub(r"\'re", " are", text)
		text = re.sub(r"\'d", " would", text)
		text = re.sub(r"\'re", " are", text)
		text = re.sub(r"won't", "will not", text)
		text = re.sub(r"can't", "cannot", text)
		text = re.sub(r"n't", " not", text)
		text = re.sub(r"n'", "ng", text)
		text = re.sub(r"'bout", "about", text)
		text = re.sub(r"'til", "until", text)

		#text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
		return text

	@staticmethod
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

	@staticmethod
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

	@staticmethod
	def _replace_unknowns(sequences, vocab, unk):
		"""
		sequences - A 2-D list of strings
		Returns
			A 2-D list of strings
		"""
		return [ [word if word in vocab else unk for word in sequence] for sequence in sequences ]

	@staticmethod
	def _encode(sequences, vocab2int):
		"""
		sequences - 2-D list of strings
		Returns
			A 2-D list of integers
		"""	
		# Convert the text to integers. 
		return [ [vocab2int[word] for word in sequence] for sequence in sequences]

	@staticmethod
	def _write_lines(path, lines):
		with open(path, "w", encoding="utf-8") as w:
			w.write( "\n".join(lines) )
