import sys
sys.path.append("../")
import loss_functions

import tensorflow as tf
import numpy as np

SEED = 1
tf.set_random_seed(SEED)
np.random.seed(SEED)


num_samples = 2
num_classes = 4
max_time = 5
	
distro_min = -2
distro_max = 2
logits = tf.random_uniform( [num_samples, max_time, num_classes], minval=distro_min, maxval=distro_max)
targets = tf.random_uniform( [num_samples, max_time], minval=0, maxval=num_classes, dtype=tf.int32 )


embed_min = -5
embed_max = 5
word_embedding_size = 8
word_embeddings = tf.random_uniform( [num_classes, word_embedding_size], minval=embed_min, maxval=embed_max)

def max_affective_content_vad_test():
	lambda_param = tf.constant(0.5, dtype=tf.float32, shape=())
	neutral_vector = tf.constant( [5., 1., 5.], dtype=tf.float32)

	emot_embedding_size = 3
	embedding_size = word_embedding_size + emot_embedding_size
	emot_embeddings = tf.random_uniform([num_classes, emot_embedding_size], minval=1, maxval=10, dtype=tf.int32)
	emot_embeddings = tf.cast(emot_embeddings, tf.float32)
	#full_embeddings = tf.concat([word_embeddings, emot_embeddings], axis=-1)

	target_lengths = tf.constant( [5, 3], dtype=tf.int32)
	weights = tf.sequence_mask(target_lengths, dtype=tf.float32)

	loss_averaging = lambda by_time, by_batch: loss_functions.max_affective_content(logits,
									targets,
									lambda_param,
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
	max_affective_content_vad_test()
