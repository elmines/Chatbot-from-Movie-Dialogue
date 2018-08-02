"""
Module for loading and namespacing corpora
"""
def read_tokens(path):
	"""
	:param path-like path: Path to file containing newline-separated sequences

	:returns: The file's text, with each entry one of the sequences
	:rtype: list(str)
	"""
	with open(path, "r", encoding="utf-8") as r:
		text = [ [token for token in line.strip().split(" ")] for line in r.readlines()]
	return text

class Corpus(object):
	"""
	Namespace for a corpus's textual tokens and their corresponding integer indices.

	:ivar list(list(str))    text: The corpus's tokens
	:ivar list(list(int)) indices: The corresponding integer indices for each token in **text**.
	"""
	def __init__(self, text, indices):
		"""
		:param list(list(str))    text: The corpus's tokens
		:param list(list(int)) indices: The corresponding integer indices for each token in **text**.
		"""
		self.text = text
		self.indices = indices

class Data(object):
	"""
	Namespace for corpora and their vocabulary
	"""
	def __init__(self, text2int, unk, **corpora_dict):
		"""
		:param dict(str,int)           text2int: Mapping from tokens to indices (**unk** must be a key)
		:param str                          unk: The unknown token
		:param dict(str,path-like) corpora_dict: Mapping from variable names to paths of text files

		For each (key, path) pair in **corpora_dict**, an instance member is added such that self.key is a Corpus object representing the text read from path.
		"""

		self._text2int = text2int
		self._unk = unk
		self._unk_int = self._text2int[self._unk]
		self._int2text = {index:word for (word, index) in self._text2int.items()}

		text_to_int = lambda sequences: [[self._text2int[token] for token in seq] for seq in sequences]
		for (name, path) in corpora_dict.items():
			text = read_tokens(path)
			indices = text_to_int(text)
			corpus = Corpus(text, indices)	
			self.__setattr__(name, corpus)

	@property
	def text2int(self):
		"""
		dict(str, int): Mapping form tokens to indices (of which **unk** is a key)
		"""
		return self._text2int

	@property
	def int2text(self):
		"""
		dict(int, str): Inverse of **text2int**.
		"""
		return self._int2text

	@property
	def unk(self):
		"""
		str: The unknown token
		"""
		return self._unk

	@property
	def unk_int(self):
		"""
		int: Convenience member for **unk**\'s integer index
		"""
		return self._unk_int
	
			
