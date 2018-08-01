"""
Module for constructing a model and either training it or inferring responses from it.
"""

#Utilities
import sys
import os
import warnings

#Computation
import tensorflow as tf
import numpy as np
SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#Data science
import pandas as pd

#Local modules
import data
import models
import training
import inference
import config


def var_dict(variables):
	"""
	:param list(tf.Variable) variables: A list of TensorFlow variables

	:returns: A dictionary mapping variable names to variable objects
	:rtype: dict(str, tf.Variable)
	"""
	return {var.name:var for var in variables}

def append_meta(wordVecs, verify_index=None):
	"""
	:param np.ndarray     wordVecs: An matrix of word embeddings
	:param int        verify_index: The index that should be used to represent the metatoken

	:returns: The word embeddings with the metatoken embeddings appended
	:rtype: np.ndarray
	"""

	embedding_size = wordVecs.shape[1] #Dynamically determine embedding size from loaded embedding file
	metatoken_embedding = np.zeros((1, embedding_size), dtype=wordVecs.dtype)
	wordVecsWithMeta = np.concatenate( (wordVecs, metatoken_embedding), axis=0 )

	metatoken_index = wordVecsWithMeta.shape[0] - 1
	if verify_index: assert metatoken_index == verify_index
	return wordVecsWithMeta


def append_eos(answers_int, eos_int):
	"""
	:param list(list(int)) answers_int: Answer sequences
	:param int                 eos_int: The token used to represent the end-of-sequence

	:returns: The answers with the EOS token appended
	:rtype: list(list(int))
	"""
	return [sequence+[eos_int] for sequence in answers_int]


class BaseExperiment(object):
	"""
	Abstract class representing an experiment to be performed
	"""

	def __init__(self, config_obj, infer=False):
		"""
		:param config.Config config_obj: The settings for the experiment
		:param bool               infer: Whether training or inference is being performed
		"""

		self.config = config_obj
		self.inference = infer

		self.model_load = self.config.model_load
		if self.inference:
			if self.model_load is None:
				raise ValueError("Must specify model_load in configuration when doing inference")
		else:
			self.train_save = self.config.train_save
			self.infer_save = self.config.infer_save
			self.train_checkpoint = None #To be defined in subclasses
			self.infer_checkpoint = None

				

		with open(config_obj.vocab, "r", encoding="utf-8") as r:
			text2int = {}
			for line in r.readlines():
				(word, index) = line.split()
				text2int[word] = int(index)

		if self.inference:
			self.data = data.Data(text2int, config_obj.unk)
		else:
			self.data = data.Data(text2int, config_obj.unk,
					train_prompts = config_obj.corpora[0],
					train_answers = config_obj.corpora[1],
					valid_prompts = config_obj.valid_corpora[0],
					valid_answers = config_obj.valid_corpora[1])

		self.vocab2int = self.data.text2int
		self.int2vocab = self.data.int2text
		self.unk_int = self.data.unk_int
		self.unk = self.data.unk

		self.metatoken = len(self.vocab2int)
		self.go_token = self.metatoken
		self.eos_token = self.metatoken
		self.pad_token = self.metatoken

		self.data.pad_int = self.pad_token #FIXME: Don't do ad-hoc adding of members to data
		
		if not self.inference:
			self.data.train_answers.indices = append_eos(self.data.train_answers.indices, self.eos_token)
			self.data.valid_answers.indices = append_eos(self.data.valid_answers.indices, self.eos_token)


	def train(self):	
		"""
		Performs the training process given the configuration passed in
		"""
		raise NotImplementedError

	def infer(self, prompts_text):
		"""
		:param list(str) prompts_text: Prompts to give the model

		:returns: A dataframe of the prompts, their corrresponding responses, and any other metadata the model provides.
		:rtype: pd.DataFrame
		"""
		if not self.inference:
			raise ValueError("Can only call infer() if model is constructed in inference mode")

		unk_int = self.data.unk_int
		vocab2int = self.data.text2int

		cleaned_prompts = [seq.strip() for seq in prompts_text]
		prompts_int = [ [vocab2int.get(token, unk_int) for token in seq.split()] for seq in cleaned_prompts]
		pad_int = self.data.pad_int

		with tf.Session() as sess:
			self.infer_checkpoint.restore(self.model_load).assert_consumed().run_restore_ops()
			sys.stderr.write("Restored model from {}\n".format(self.model_load))
			beam_outputs = inference.infer(sess, self.model, prompts_int, self.infer_feeds, self.model.beams, pad_int, batch_size=self.config.infer_batch_size)

		int2vocab = self.data.int2text
		beam_width = len(beam_outputs[0][0][:])

		out_frame = pd.DataFrame({"prompts": prompts_text})
		for j in range(beam_width):
			label = "beams_" + str(j)
			column = []
			for i in range(len(beam_outputs)):
				beam = beam_outputs[i][:, j] #jth beam for the ith sample
				beam_text = " ".join([int2vocab[token] for token in beam if token != pad_int])
				column.append(beam_text)
			out_frame[label] = column
		return out_frame

	def save_fn(self, sess):
		"""
		:param tf.Session sess: The session in which the model is being trained

		Saves the training and inference graphs of the model
		"""
		act_train_prefix = self.train_checkpoint.save(self.train_save, sess)
		print("Saved training graph to {}".format(act_train_prefix))
		act_infer_prefix = self.infer_checkpoint.save(self.infer_save, sess)
		print("Saved inference graph to {}".format(act_infer_prefix))

class VADExp(BaseExperiment):
	"""
	Experiment involving embeddings with VAD values appended to the end
	"""
	def __init__(self, config_obj, infer=False):
		"""
		:param config.Config config_obj: The settings for the experiment
		:param          bool      infer: Whether to perform training or inference
		"""
		BaseExperiment.__init__(self, config_obj, infer)

		full_embeddings = np.load(config_obj.embeddings).astype(np.float32)
		self.wordVecsWithMeta = append_meta(full_embeddings, self.metatoken)
		
		tf.reset_default_graph()
		embeddings_var = tf.constant(self.wordVecsWithMeta, name="embeddings")
		output_layer = tf.layers.Dense(len(self.wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)

		self.model = models.VADAppended(embeddings_var, self.go_token, self.eos_token, self.config, output_layer=output_layer, affect_strength = 0.2, infer=self.inference)

		self.train_feeds = {self.model.keep_prob: 0.75}
		self.infer_feeds = {self.model.keep_prob: 1}

		#Set variables to be saved
		self.train_checkpoint = None
		if not self.inference:
			global_dict = var_dict( tf.global_variables() )
			self.train_checkpoint = tf.train.Checkpoint(**global_dict)
		train_dict = var_dict(tf.trainable_variables())
		self.infer_checkpoint = tf.train.Checkpoint(**train_dict)


	def train(self, train_affect=False):
		"""
		Performs the training process given the configuration passed in

		:param bool train_affect: Whether to train using cross-entropy or affective loss
		"""
		if self.inference:
			raise ValueError("Tried to train a model in inference mode.")
		xent_epochs = self.config.max_epochs 

		trainer = training.Trainer(self.save_fn, max_epochs=xent_epochs, max_stalled_steps=5)

		with tf.Session() as sess:
			if self.model_load:
				warnings.warn("You are reloading a model for training. This feature"
						" is still not fully implemented. It restores the state"
						" of the model variables and optimizer but not the number"
						" of stalled steps, the validation cost record, or the"
						" state of the shuffled corpora")
				self.train_checkpoint.restore(self.model_load).assert_consumed().run_restore_ops()
				print("Restored model at {}".format(self.model_load))
			else:
				sess.run(tf.global_variables_initializer())


			training.training_loop(sess, self.model, trainer, self.data, self.train_feeds, self.infer_feeds,
						train_batch_size=self.config.train_batch_size,
						valid_batch_size=self.config.infer_batch_size,
						min_epochs_before_validation=1)

			if train_affect:
				affect_epochs = (trainer.epochs_completed // 4) + 1*(trainer.epochs_completed < 4)
				total_epochs = trainer.epochs_completed + affect_epochs
				train_feeds[self.model.train_affect] = True
				print("Switching from cross-entropy to maximum affective content . . .")

				affect_trainer = training.Trainer(self.checkpoint_best, self.checkpoint_latest, save_fn, epochs_completed=trainer.epochs_completed,
						max_epochs=total_epochs, saver=trainer.saver,
						best_valid_cost = trainer.best_valid_cost)
				training.training_loop(sess, self.model, affect_trainer, self.data, self.train_feeds, self.valid_feeds,
						train_batch_size=self.config.train_batch_size,
						valid_batch_size=self.config.infer_batch_size)



class DistrExp(BaseExperiment):
	"""
	An experiment using distributed affective embeddings.
	"""
	def __init__(self, config_obj, infer=False):
		"""
		:param config.Config config_obj: The settings for the experiment
		:param          bool      infer: Whether to perform training or inference
		"""
		BaseExperiment.__init__(self, config_obj, infer)


		full_embeddings = np.load(config_obj.embeddings).astype(np.float32)
		self.wordVecsWithMeta = append_meta(full_embeddings, self.metatoken)
		
		tf.reset_default_graph()
		embeddings_var = tf.constant(self.wordVecsWithMeta, name="embeddings")
		output_layer = tf.layers.Dense(len(self.wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)
		self.model = models.Aff2Vec(enc_embeddings=embeddings_var, dec_embeddings=embeddings_var, go_token=self.go_token, eos_token=self.eos_token, config=self.config, output_layer=output_layer, infer=self.inference)

		self.train_feeds = {self.model.keep_prob: 0.75}
		self.infer_feeds = {self.model.keep_prob: 1}

		#Set variables to be saved
		self.train_checkpoint = None
		if not self.inference:
			global_dict = var_dict( tf.global_variables() )
			self.train_checkpoint = tf.train.Checkpoint(**global_dict)
		train_dict = var_dict(tf.trainable_variables())
		self.infer_checkpoint = tf.train.Checkpoint(**train_dict)

	def train(self):
		"""
		Performs the training process given the configuration passed in
		"""
		if self.inference:
			raise ValueError("Tried to train a model in inference mode.")
		xent_epochs = self.config.max_epochs
		trainer = training.Trainer(self.save_fn, max_epochs=xent_epochs, max_stalled_steps=5)

		with tf.Session() as sess:
			if self.model_load:
				self.train_checkpoint.restore(self.model_load).assert_consumed().run_restore_ops()
				warnings.warn("You are reloading a model for training. This feature"
						" is still not fully implemented. It restores the state"
						" of the model variables and optimizer but not the number"
						" of stalled steps, the validation cost record, or the"
						" state of the shuffled corpora")
				print("Restored model at {}".format(self.model_load))
			else:
				sess.run(tf.global_variables_initializer())
			training.training_loop(sess, self.model, trainer, self.data, self.train_feeds, self.infer_feeds,
						train_batch_size=self.config.train_batch_size,
						valid_batch_size=self.config.infer_batch_size,
						min_epochs_before_validation=1)

