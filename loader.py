import os
import numpy as np

#Local modules
import embeddings

def read_tokens(path):
	with open(path, "r", encoding="utf-8") as r:
		text = [ [token for token in line.strip().split(" ")] for line in r.readlines()]
	return text

class Loader(object):

	def __init__(self, data_dir="corpora/", w2vec_path="word_Vecs.npy", regenerate=True, new_embedding_size=1024):
	
		var_names = ["train_prompts", "train_answers", "valid_prompts", "valid_answers", "vocab"]
		file_names = [os.path.join(data_dir, var_name + ".txt") for var_name in var_names]
	
		self.train_prompts = read_tokens(file_names[0])
		self.train_answers = read_tokens(file_names[1])
		self.valid_prompts = read_tokens(file_names[2])
		self.valid_answers = read_tokens(file_names[3])
	
		vocab = read_tokens(file_names[4])
		self.vocab2int = {pair[0]:int(pair[1]) for pair in vocab}
		self.int2vocab = {index:word for (word, index) in self.vocab2int.items()}

		self.unk_int = 0
		self.unk = self.int2vocab[self.unk_int] #FIXME: Don't rely on the magic number 0	
	

		text_to_int = lambda sequences: [ [self.vocab2int[token] for token in seq] for seq in sequences]
		self.train_prompts_int = text_to_int(self.train_prompts)
		self.train_answers_int = text_to_int(self.train_answers)
		self.valid_prompts_int = text_to_int(self.valid_prompts)
		self.valid_answers_int = text_to_int(self.valid_answers)

		self.full_text = self.train_prompts+self.train_answers+self.valid_prompts+self.valid_answers

		if regenerate:
			self.w2vec_embeddings = embeddings.w2vec(w2vec_path, self.full_text, self.vocab2int, embedding_size=new_embedding_size)
		else:
			self.w2vec_embeddings = np.load(w2vec_path)
		self.w2vec_embeddings = self.w2vec_embeddings.astype(np.float32)

		self.vad_embeddings = None #Just declare the field
		self.counter_embeddings = None
		self.retro_embeddings = None

	def load_vad(self, vad_vec_path="word_Vecs_VAD.npy", regenerate=True, verbose=True):
		"""
		Returns Numpy array"
		"""
		if regenerate:
			self.vad_embeddings  = embeddings.appended_vad(vad_vec_path, self.w2vec_embeddings, self.vocab2int, exclude=[self.unk])
		else:
			self.vad_embeddings = np.load(vad_vec_path)
		self.vad_embeddings = self.vad_embeddings.astype(np.float32)

		return self.vad_embeddings
	
