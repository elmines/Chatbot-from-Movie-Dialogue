import sys

#sys.path.insert(0, "/users/emines/.local/lib/python3.5/site-packages/")

sys.path.append("/users/emines/.local/lib/python3.5/site-packages/")

import tensorflow as tf
sys.stderr.write("TensorFlow {}\n".format(tf.VERSION))

import numpy as np
import os
import time



SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#Local modules
import loader
import models
import training
import tf_collections


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


def vad_appended_experiment(regen_embeddings=False, data_dir="corpora/", w2vec_path="word_Vecs.npy", vad_vec_path = "word_Vecs_VAD.npy"):
	data = loader.Loader(data_dir, w2vec_path, regenerate=regen_embeddings)

	train_prompts_int = data.train_prompts_int
	train_answers_int = data.train_answers_int
	valid_prompts_int = data.valid_prompts_int
	valid_answers_int = data.valid_answers_int
	vocab2int = data.vocab2int
	int2vocab = data.int2vocab
	unk_int = data.unk_int
	unk = data.unk
	
	full_embeddings = data.load_vad(vad_vec_path, regenerate=regen_embeddings)
	(wordVecsWithMeta, metatoken) = word_vecs_with_meta(full_embeddings)
	go_token = metatoken
	eos_token = metatoken
	pad_token = metatoken

	train_answers_int = append_eos(train_answers_int, eos_token)
	valid_answers_int = append_eos(valid_answers_int, eos_token)

	tf.reset_default_graph()
	data_placeholders = models.create_placeholders()
	output_layer = tf.layers.Dense(len(wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)
	model = models.VADAppended(data_placeholders, wordVecsWithMeta, go_token, eos_token, output_layer=output_layer)

	#Used to make unique directories, not to identify when a model is saved
	time_string = time.strftime("%b%d_%H:%M:%S")
	checkpoint_dir = os.path.join("checkpoints", time_string)
	if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
	checkpoint_best = str(checkpoint_dir) + "/" + "best_model.ckpt" 
	checkpoint_latest = str(checkpoint_dir) + "/" + "latest_model.ckpt"


	datasets = tf_collections.Datasets(train_prompts_int=train_prompts_int,
						train_answers_int=train_answers_int,
						valid_prompts_int=valid_prompts_int,
						valid_answers_int=valid_answers_int
					)
	trainer = training.Trainer(checkpoint_best, checkpoint_latest, max_stalled_steps = 5)
	text_data = tf_collections.TextData(prompts_int2vocab=int2vocab, answers_int2vocab=int2vocab, unk_int=unk_int, eos_int=eos_token, pad_int=pad_token)

	xent_epochs = 40
	train_feeds = {model.keep_prob: 0.75}
	valid_feeds = {model.keep_prob: 1}

	with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
		sess.run(tf.global_variables_initializer())
		epochs_completed, best_valid_loss = training.training_loop(sess, model, trainer, datasets, text_data, train_feeds, valid_feeds)

		affect_epochs = (trainer.epochs_completed // 5) + 1*(trainer.epochs_completed < 5)

		affect_trainer = Trainer(checkpoint_best, checkpoint_latest, epochs_completed=trainer.epochs_completed,
						max_epochs=trainer.epochs_completed+affect_epochs, saver=trainer.saver,
						best_valid_cost = trainer.best_valid_cost)

		
		epochs_completed, best_valid_loss = training.training_loop(sess, model, affect_trainer, datasets, text_data, train_feeds, valid_feeds)	

	

if __name__ == "__main__":
	if len(sys.argv) > 1:
		if sys.argv[1] == "--embeddings":
			gen_embeddings()
	else:
		vad_appended_experiment(regen_embeddings=False)
