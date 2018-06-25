import sys
sys.path.append("../")
import loss_functions

import tensorflow as tf
import numpy as np

def test_average():
	num_samples = 3
	max_time = 5
	embedding_size = 10
	np.random.seed(1)

	batch = np.random.randn(num_samples, max_time, embedding_size).astype(np.float32)
	print(batch)

	averages = loss_functions.embed_predicted_timestep(batch)

	with tf.Session() as sess:
		[averages_out] = sess.run([averages])
	print(averages_out)

def reduce_gather_test():
	num_samples = 2
	max_time = 5
	num_classes = 6
	
	logits = np.random.randn(num_samples, max_time, num_classes).astype(np.float32)
	targets = np.random.randint(num_classes, size=(num_samples, max_time))
	
	print("Logits")
	print(logits)
	print("targets")
	print(targets)
	
	
	gathered_new = loss_functions.reduce_gather(logits, targets)
	
	with tf.Session() as sess:
		[gathered_new_out] = sess.run( [gathered_new] )
	print("gathered")
	print(gathered_new_out)

SEED = 1
tf.set_random_seed(SEED)
np.random.seed(SEED)

#DATASET DIMENSIONS
num_samples = 2
num_classes = 4
max_time_enc = 3
max_time = 5
	
#EMBEDDINGS
embed_min = -5
embed_max = 5
word_embedding_size = 8
word_embeddings = tf.random_uniform( [num_classes, word_embedding_size], minval=embed_min, maxval=embed_max)
emot_embedding_size = 3
embedding_size = word_embedding_size + emot_embedding_size
emot_embeddings = tf.random_uniform([num_classes, emot_embedding_size], minval=1, maxval=10, dtype=tf.int32)
emot_embeddings = tf.cast(emot_embeddings, tf.float32)


distro_min = -2.
distro_max = 2.
logits = tf.random_uniform( [num_samples, max_time, num_classes], minval=distro_min, maxval=distro_max, dtype=tf.float32)
targets = tf.random_uniform( [num_samples, max_time], minval=0, maxval=num_classes, dtype=tf.int32 )
target_lengths = tf.constant( [3, 5], dtype=tf.int32)
weights = tf.sequence_mask(target_lengths, dtype=tf.float32)

enc_embed_input = tf.nn.embedding_lookup(emot_embeddings, tf.random_uniform([num_samples, max_time_enc], minval=0, maxval=num_classes, dtype=tf.int32))


lambda_param = tf.constant(0.4, dtype=tf.float32, shape=())

def _affect_dissonance_wrapper(maximum, by_time, by_batch):
	if maximum:
		return loss_functions.max_affective_dissonance(lambda_param,
								logits,
								targets,
								enc_embed_input,
								emot_embeddings, 
								weights=weights,
								average_across_timesteps=by_time,
								average_across_batch=by_batch
								)
	else:
		return loss_functions.min_affective_dissonance(lambda_param,
								logits,
								targets,
								enc_embed_input,
								emot_embeddings, 
								weights=weights,
								average_across_timesteps=by_time,
								average_across_batch=by_batch
								)

								

def affective_dissonance_test():
	function_names = ["maximum_affective_content", "minimum_affective_content"]
	function_vals = [True, False]	

	sess = tf.InteractiveSession()
	inputs = [lambda_param, logits, targets, weights, emot_embeddings, enc_embed_input]
	names = "lambda_param, logits, targets, weights, emot_embeddings, enc_embed_input".split(", ")


	for i in range(len(function_names)):
		print(function_names[i])
		loss_fetches = [_affect_dissonance_wrapper(function_vals[i], False, False),
				_affect_dissonance_wrapper(function_vals[i], False, True),
				_affect_dissonance_wrapper(function_vals[i], True, False),
				_affect_dissonance_wrapper(function_vals[i], True, True)]

		loss_names = "by-token, by-timestep, by-batch, averaged".split(", ")
		output_names = names+loss_names

		outputs = sess.run( inputs+loss_fetches )

		for i in range(len(outputs)):
			print("\t{}".format(output_names[i]))
			print("\t{}".format(outputs[i]))


	sess.close()

def max_affective_content_vad_test():
	neutral_vector = tf.constant( [5., 1., 5.], dtype=tf.float32)



	loss_averaging = lambda by_time, by_batch: loss_functions.max_affective_content(lambda_param,
									logits,
									targets,
									emot_embeddings,
									neutral_vector,
									weights=weights,
									average_across_batch=by_batch,
									average_across_timesteps=by_time)

	losses = loss_averaging(False, False)
	avg_time_loss = loss_averaging(False, True)
	avg_sample_loss =  loss_averaging(True, False)
	avg_loss = loss_averaging(True, True)


	fetch_args = [logits, targets, weights, emot_embeddings, losses, avg_sample_loss, avg_time_loss, avg_loss]
	output_names = "logits, targets, weights, emot_embeddings, losses, avg_sample_loss, avg_time_loss, avg_loss".split(", ")
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		outputs = sess.run ( fetch_args )

	for (output_name, output) in zip(output_names, outputs):
		print(output_name)
		print(output)

if __name__ == "__main__":
	affective_dissonance_test()
	#test_average()
	#reduce_gather_test()
	max_affective_content_vad_test()
