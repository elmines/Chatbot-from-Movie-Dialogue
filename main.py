import sys
import tensorflow as tf
sys.stderr.write("TensorFlow {}\n".format(tf.VERSION))

import numpy as np
import os
import time



SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#Local modules
import embeddings
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


def read_tokens(path):
	with open(path, "r", encoding="utf-8") as r:
		text = [ [token for token in line.strip().split(" ")] for line in r.readlines()]
	return text

def load_data(data_dir):
	"""
	For now, data_dir must have
		- train_prompts.txt
		- train_answers.txt
		- valid_prompts.txt
		- valid_answers.txt
		- vocab.txt

	Returns
		(train_prompts, train_answers, valid_prompts, valid_answers, vocab2int)
	"""

	var_names = ["train_prompts", "train_answers", "valid_prompts", "valid_answers", "vocab"]
	file_names = [os.path.join(data_dir, var_name + ".txt") for var_name in var_names]

	train_prompts = read_tokens(file_names[0])
	train_answers = read_tokens(file_names[1])
	valid_prompts = read_tokens(file_names[2])
	valid_answers = read_tokens(file_names[3])

	vocab = read_tokens(file_names[4])
	vocab2int = {pair[0]:int(pair[1]) for pair in vocab}

	return (train_prompts, train_answers, valid_prompts, valid_answers, vocab2int)
	

def append_eos(answers_int, eos_int):
	return [sequence+[eos_int] for sequence in answers_int]


def vad_appended_experiment(regen_embeddings=False, data_dir="corpora/", w2vec_path="word_Vecs.npy", vad_vec_path = "word_Vecs_VAD.npy"):
	data = load_data(data_dir)
	train_prompts = data[0]
	train_answers = data[1]
	valid_prompts = data[2]
	valid_answers = data[3]
	vocab2int = data[4]
	int2vocab = {index:key for key, index in vocab2int.items()}
	
	unk_int = 0
	unk = int2vocab[unk_int] #FIXME: Don't rely on the magic number 0	
	

	text_to_int = lambda sequences: [ [vocab2int[token] for token in seq] for seq in sequences]
	train_prompts_int = text_to_int(train_prompts)
	train_answers_int = text_to_int(train_answers)
	valid_prompts_int = text_to_int(valid_prompts)
	valid_answers_int = text_to_int(valid_answers)
	
	if regen_embeddings:
		full_text = train_prompts+train_answers+valid_prompts+valid_answers
		w2vec_embeddings = embeddings.w2vec(w2vec_path, full_text, vocab2int, embedding_size=1024)
		full_embeddings = embeddings.appended_vad(vad_vec_path, w2vec_embeddings, vocab2int, exclude=[unk])
	else:
		full_embeddings = np.load(vad_vec_path)	

	full_embeddings = full_embeddings.astype(np.float32)

	(wordVecsWithMeta, metatoken) = word_vecs_with_meta(full_embeddings)
	go_token = metatoken
	eos_token = metatoken
	pad_token = metatoken

	train_answers_int = append_eos(train_answers_int, eos_token)
	valid_answers_int = append_eos(valid_answers_int, eos_token)


	tf.reset_default_graph()
	with tf.device("/cpu:0"):
		data_placeholders = models.create_placeholders()
		output_layer = tf.layers.Dense(len(wordVecsWithMeta),bias_initializer=tf.zeros_initializer(),activation=tf.nn.relu)
		model = models.VADAppended(data_placeholders, full_embeddings, go_token, eos_token, output_layer=output_layer)

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

	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		epochs_completed, best_valid_loss = training.training_loop(sess, model, trainer, datasets, text_data, train_feeds, valid_feeds)

		affect_epochs = (trainer.epochs_completed // 5) + 1*(trainer.epochs_completed < 5)

		affect_trainer = Trainer(checkpoint_best, checkpoint_latest, epochs_completed=trainer.epochs_completed,
						max_epochs=trainer.epochs_completed+affect_epochs, saver=trainer.saver,
						best_valid_cost = trainer.best_valid_cost)

		
		epochs_completed, best_valid_loss = training.training_loop(sess, model, affect_trainer, datasets, text_data, train_feeds, valid_feeds)	

	

if __name__ == "__main__":
	vad_appended_experiment(regen_embeddings=False)
