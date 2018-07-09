#Utilities
import sys
import os
from enum import Enum
import time

#Computation
import tensorflow as tf
import numpy as np

#Local modules
import loader
import models
import training
import tf_collections
import test

from preprocessing.corpus import pre_clean_seq
from preprocessing.corpus import post_clean_seq

def write_vars(path, variables):
	with open(path, "w", encoding="utf-8") as out:
		for var in variables:
			out.write("{}\n".format(var))	

def append_meta(wordVecs, verify_index=None):
	"""
	wordVecs - an np array of word embeddings
	:param int verify_index: The index that should be used to represent the metatoken
	Returns
		(np array with the metatoken embeddings appended, the index of the metatoken embedding)
	"""

	embedding_size = wordVecs.shape[1] #Dynamically determine embedding size from loaded embedding file
	metatoken_embedding = np.zeros((1, embedding_size), dtype=wordVecs.dtype)
	wordVecsWithMeta = np.concatenate( (wordVecs, metatoken_embedding), axis=0 )

	metatoken_index = wordVecsWithMeta.shape[0] - 1
	if verify_index: assert metatoken_index == verify_index
	return wordVecsWithMeta


def append_eos(answers_int, eos_int):
	return [sequence+[eos_int] for sequence in answers_int]

def gen_embeddings(vad=True, counter=True, retro=True, data_dir="corpora/", w2vec_path="word_Vecs.npy", vad_vec_path="word_Vecs_VAD.npy", counterfit_path="word_Vecs_counterfit_affect.npy", retrofit_path="word_Vecs_retrofit_affect.npy", verbose=True):
	data_loader = loader.Loader(data_dir, w2vec_path, regenerate=True) #Loads the vanilla Word2Vec embeddings
	if vad:     data_loader.load_vad(vad_vec_path, regenerate=True)
	if counter: data_loader.load_counterfit(counterfit_path, "./w2v_counterfit_append_affect.bin", regenerate=True)
	if retro:   data_loader.load_retrofit(retrofit_path, "./w2v_retrofit_append_affect.bin", regenerate=True)

class ExpState(Enum):
	NEW = 1
	CONT_TRAIN = 2
	QUERY = 3


class Experiment(object):

	def __init__(self, regenerate=False, data_dir="corpora/", w2vec_path="word_Vecs.npy", exp_state=ExpState.NEW):
		"""
		:param str restore_path: TensorFlow checkpoint from which to restore a model (only applicable if exp_state != ExpState.NEW)
		"""

		self.exp_state = exp_state

		if self.exp_state != ExpState.NEW:
			regenerate = False
		self.data = loader.Loader(data_dir, w2vec_path, regenerate=regenerate)
		self.vocab2int = self.data.vocab2int
		self.int2vocab = self.data.int2vocab
		self.unk_int = self.data.unk_int
		self.unk = self.data.unk

		self.metatoken = len(self.vocab2int)
		self.go_token = self.metatoken
		self.eos_token = self.metatoken
		self.pad_token = self.metatoken
		

		self.text_data = tf_collections.TextData(prompts_int2vocab=self.int2vocab, answers_int2vocab=self.int2vocab,
						unk_int=self.unk_int, eos_int=self.eos_token, pad_int=self.pad_token)


		self.train_prompts_int = self.data.train_prompts_int
		self.train_answers_int = append_eos(self.data.train_answers_int, self.eos_token)
		self.valid_prompts_int = self.data.valid_prompts_int
		self.valid_answers_int = append_eos(self.data.valid_answers_int, self.eos_token)

		self.datasets = tf_collections.Datasets(train_prompts_int=self.train_prompts_int,
						train_answers_int=self.train_answers_int,
						valid_prompts_int=self.valid_prompts_int,
						valid_answers_int=self.valid_answers_int
					)

		if self.exp_state != ExpState.QUERY:
			#Used to make unique directories, not to identify when a model is saved
			time_string = time.strftime("%b%d_%H:%M:%S")
			self.checkpoint_dir = os.path.join("checkpoints", time_string)
			if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
			self.checkpoint_best = str(self.checkpoint_dir) + "/" + "best_model.ckpt" 
			self.checkpoint_latest = str(self.checkpoint_dir) + "/" + "latest_model.ckpt"
			sys.stderr.write("Writing all new model files to {}\n".format(self.checkpoint_dir))


	def train(self):	
		raise NotImplementedError

	def infer(self, restore_path, prompts_text):
		raise NotImplementedError


	def infer(self, restore_path, prompts_text, pre_clean=False):
		"""
		:param str       restore_path: TensorFlow checkpoint from which to restore the model
		:param list(str) prompts_text: Prompts to give the model
		:param boolean      pre_clean: Whether prompts_text needs to be cleaned

		Writes the cleaned prompts and their corresponding response(s) to standard output
		"""

		write_vars("global.txt", tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
		write_vars("model.txt", tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
		write_vars("trainable.txt", tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
		sys.exit(0)

		unk_int = self.data.unk_int
		vocab2int = self.data.vocab2int

		cleaned_prompts = [pre_clean_seq(seq) for seq in prompts_text] if pre_clean else [seq.strip() for seq in prompts_text]
		prompts_int = [ [vocab2int.get(token, unk_int) for token in seq.split()] for seq in cleaned_prompts]
		pad_int = self.text_data.pad_int

		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, restore_path)
			beam_outputs = test.infer(sess, self.model, prompts_int, self.infer_feeds, self.model.beams, pad_int, batch_size = 32)


		int2vocab = self.data.int2vocab
		lines = []
		beam_width = len(beam_outputs[0][0][:])
		for i in range(len(beam_outputs)):
			lines.append(cleaned_prompts[i] + "\n")
			for j in range(beam_width):
				beam = beam_outputs[i][:][j] #jth beam for the ith sample
				beam_text = post_clean_seq( " ".join([int2vocab[token] for token in beam if token != pad_int]) )
				lines.append( "\t{}\n".format(beam_text) )

		for line in lines:
			sys.stdout.write(line)

class VADExp(Experiment):
	def __init__(self, regenerate=False, data_dir="corpora/", w2vec_path="word_Vecs.npy", vad_vec_path="word_Vecs_VAD.npy", exp_state=ExpState.NEW):
		Experiment.__init__(self, regenerate, data_dir, w2vec_path, exp_state)

		full_embeddings = self.data.load_vad(vad_vec_path, regenerate=regenerate)
		self.wordVecsWithMeta = append_meta(full_embeddings, self.metatoken)

		tf.reset_default_graph()
		embeddings_var = tf.constant(self.wordVecsWithMeta, name="embeddings")
		output_layer = tf.layers.Dense(len(self.wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)
		self.model = models.VADAppended(embeddings_var, self.go_token, self.eos_token, output_layer=output_layer, affect_strength = 0.2, beam_width=10)

		self.train_feeds = {self.model.keep_prob: 0.75}
		self.infer_feeds = {self.model.keep_prob: 1}



	def train(self, train_affect=False):
		if self.exp_state == ExpState.QUERY:
			raise ValueError("Tried to train a model in query mode.")
		xent_epochs = 15 

		trainer = training.Trainer(self.checkpoint_best, self.checkpoint_latest, max_epochs=xent_epochs, max_stalled_steps=5)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			training.training_loop(sess, self.model, trainer, self.datasets, self.text_data, self.train_feeds, self.infer_feeds, min_epochs_before_validation=1)

			if train_affect:
				affect_epochs = (trainer.epochs_completed // 4) + 1*(trainer.epochs_completed < 4)
				total_epochs = trainer.epochs_completed + affect_epochs
				train_feeds[self.model.train_affect] = True
				print("Switching from cross-entropy to maximum affective content . . .")

				affect_trainer = training.Trainer(self.checkpoint_best, self.checkpoint_latest, epochs_completed=trainer.epochs_completed,
						max_epochs=total_epochs, saver=trainer.saver,
						best_valid_cost = trainer.best_valid_cost)

				training.training_loop(sess, self.model, affect_trainer, self.datasets, self.text_data, self.train_feeds, self.infer_feeds)	


class Aff2VecExp(Experiment):
	def __init__(self, regenerate=False, data_dir="corpora/", w2vec_path="word_Vecs.npy", exp_state=ExpState.NEW, counterfit=True):
		Experiment.__init__(self, regenerate, data_dir, w2vec_path, exp_state, restore_path)

		if counterfit:
			full_embeddings = self.data.load_counterfit("word_Vecs_counterfit_affect.npy", "./w2v_counterfit_append_affect.bin", regenerate=regenerate)
		else:
			full_embeddings = self.data.load_retrofit("word_Vecs_retrofit_affect.npy", "./w2v_retrofit_append_affect.bin", regenerate=regenerate)
		self.wordVecsWithMeta = append_meta(full_embeddings, self.metatoken)
		
		tf.reset_default_graph()
		embeddings_var = tf.constant(self.wordVecsWithMeta, name="embeddings")
		#embeddings_var = tf.Variable(self.wordVecsWithMeta, trainable=False, name="embeddings")
		output_layer = tf.layers.Dense(len(self.wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)
		self.model = models.Aff2Vec(embeddings_var, embeddings_var, self.go_token, self.eos_token, output_layer=output_layer, beam_width=10)

		self.train_feeds = {self.model.keep_prob: 0.75}
		self.infer_feeds = {self.model.keep_prob: 1}

	def train(self):
		if self.exp_state == ExpState.QUERY:
			raise ValueError("Tried to train a model in query mode.")
		xent_epochs = 15
		trainer = training.Trainer(self.checkpoint_best, self.checkpoint_latest, max_epochs=xent_epochs, max_stalled_steps=5)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			training.training_loop(sess, self.model, trainer, self.datasets, self.text_data, self.train_feeds, self.infer_feeds, min_epochs_before_validation=1)

