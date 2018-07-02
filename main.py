import sys
import tensorflow as tf
sys.stderr.write("TensorFlow {}\n".format(tf.VERSION))
import numpy as np
import os
import time

import argparse


#Local modules
import loader
import models
import training
import tf_collections

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

def create_parser():
	parser = argparse.ArgumentParser(description="Train an affective neural dialog generation model")

	parser.add_argument("--embeddings-only", action="store_true", help="Just generate the embeddings for all the models and exit")

	parser.add_argument("--regen-embeddings", action="store_true", help="Regenerate affective embeddings prior to training")

	parser.add_argument("--vad", action="store_true", help="Train the model with VAD values appended to Word2Vec embeddings")
	parser.add_argument("--counter", action="store_true", help="Train the model with counterfitted embeddings")
	parser.add_argument("--retro", action="store_true", help="Train the model with retrofitted embeddings")

	return parser

def word_vecs_with_meta(wordVecs):
	"""
	wordVecs - an np array of word embeddings
	Returns
		(np array with the metatoken embeddings appended, the index of the metatoken embedding)
	"""
	embedding_size = wordVecs.shape[1] #Dynamically determine embedding size from loaded embedding file
	metatoken_embedding = np.zeros((1, embedding_size), dtype=wordVecs.dtype)
	wordVecsWithMeta = np.concatenate( (wordVecs, metatoken_embedding), axis=0 )
	return wordVecsWithMeta, wordVecsWithMeta.shape[0]-1


def append_eos(answers_int, eos_int):
	return [sequence+[eos_int] for sequence in answers_int]

def gen_embeddings(data_dir="corpora/", w2vec_path="word_Vecs.npy", vad_vec_path="word_Vecs_VAD.npy", verbose=True):
	data_loader = loader.Loader(data_dir, w2vec_path, regenerate=True) #Loads the vanilla Word2Vec embeddings
	data_loader.load_vad(vad_vec_path, regenerate=True)


class Experiment(object):

	def __init__(self, regenerate=False, data_dir="corpora/", w2vec_path="word_Vecs.npy"):
		self.regenerate = regenerate
		self.data = loader.Loader(data_dir, w2vec_path, regenerate=regenerate)

		self.train_prompts_int = self.data.train_prompts_int
		self.train_answers_int = self.data.train_answers_int
		self.valid_prompts_int = self.data.valid_prompts_int
		self.valid_answers_int = self.data.valid_answers_int
		self.vocab2int = self.data.vocab2int
		self.int2vocab = self.data.int2vocab
		self.unk_int = self.data.unk_int
		self.unk = self.data.unk

	def prep_data(self):

		self.train_answers_int = append_eos(self.train_answers_int, self.eos_token)
		self.valid_answers_int = append_eos(self.valid_answers_int, self.eos_token)

		self.datasets = tf_collections.Datasets(train_prompts_int=self.train_prompts_int,
						train_answers_int=self.train_answers_int,
						valid_prompts_int=self.valid_prompts_int,
						valid_answers_int=self.valid_answers_int
					)

		#Used to make unique directories, not to identify when a model is saved
		time_string = time.strftime("%b%d_%H:%M:%S")
		self.checkpoint_dir = os.path.join("checkpoints", time_string)
		if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
		self.checkpoint_best = str(self.checkpoint_dir) + "/" + "best_model.ckpt" 
		self.checkpoint_latest = str(self.checkpoint_dir) + "/" + "latest_model.ckpt"
		sys.stderr.write("Writing all model files to {}\n".format(self.checkpoint_dir))

	def run(self):	
		raise NotImplementedError

class VADExp(Experiment):
	def __init__(self, regenerate=False, data_dir="corpora/", w2vec_path="word_Vecs.npy", vad_vec_path="word_Vecs_VAD.npy"):
		Experiment.__init__(self, regenerate, data_dir, w2vec_path)

		full_embeddings = self.data.load_vad(vad_vec_path, regenerate=regenerate)
		(self.wordVecsWithMeta, metatoken) = word_vecs_with_meta(full_embeddings)
		self.go_token = metatoken
		self.eos_token = metatoken
		self.pad_token = metatoken

		self.prep_data()

	def run(self):
		tf.reset_default_graph()
		data_placeholders = models.create_placeholders()
		output_layer = tf.layers.Dense(len(self.wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)
		model = models.VADAppended(data_placeholders, self.wordVecsWithMeta, self.go_token, self.eos_token, output_layer=output_layer, affect_strength = 0.2)


		xent_epochs = 8
		train_feeds = {model.keep_prob: 0.75}
		valid_feeds = {model.keep_prob: 1}

		trainer = training.Trainer(self.checkpoint_best, self.checkpoint_latest, max_epochs=xent_epochs)
		text_data = tf_collections.TextData(prompts_int2vocab=self.int2vocab,
						answers_int2vocab=self.int2vocab,
						unk_int=self.unk_int, eos_int=self.eos_token, pad_int=self.pad_token)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			training.training_loop(sess, model, trainer, self.datasets, text_data, train_feeds, valid_feeds, min_epochs_before_validation=2)

			affect_epochs = (trainer.epochs_completed // 4) + 1*(trainer.epochs_completed < 4)
			total_epochs = trainer.epochs_completed + affect_epochs
			train_feeds[model.train_affect] = True
			sys.stderr.write("Switching from cross-entropy to maximum affective content . . .\n")

			affect_trainer = training.Trainer(self.checkpoint_best, self.checkpoint_latest, epochs_completed=trainer.epochs_completed,
						max_epochs=total_epochs, saver=trainer.saver,
						best_valid_cost = trainer.best_valid_cost)

			training.training_loop(sess, model, affect_trainer, self.datasets, text_data, train_feeds, valid_feeds)	

	
if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()
	if args.embeddings_only:
		gen_embeddings()
		sys.exit(0)

	regenerate = args.regen_embeddings
	if args.vad:
		vad_exp = VADExp(regenerate)
		vad_exp.run()
	elif args.counter:
		pass
	elif args.retro:
		pass
