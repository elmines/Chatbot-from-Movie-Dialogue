import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")
import metrics

SEED = 1
tf.set_random_seed(SEED)
np.random.seed(SEED)

def perplexity_test():
	num_classes = 5
	num_samples = 3

	max_sequence_length = 20
	target_sequence_lengths = [np.random.randint(1, max_sequence_length+1) for _ in range(num_samples)]
	actual_sequence_lengths = [np.random.randint(1, max_sequence_length+1) for _ in range(num_samples)]

	targets = np.zeros( (num_samples, max_sequence_length), dtype=np.int32 )
	predictions = np.zeros( (num_samples, max_sequence_length, num_classes), dtype=np.float32)
	for i in range(num_samples):
		predictions[i , :actual_sequence_lengths[i] ] = np.random.normal(loc=20, size = (actual_sequence_lengths[i], num_classes) )
		targets[i, :target_sequence_lengths[i] ] = np.random.randint(num_classes, size = (target_sequence_lengths[i]) )
		
	eval_mask = tf.sequence_mask(target_sequence_lengths, maxlen=max_sequence_length, dtype=tf.float32)

	#print("TARGETS")
	#print(targets)
	#print()

	#print("PREDICTIONS")
	#print(predictions)
	#print()

	#print("MASK")
	#print(eval_mask)

	tf_targets = tf.constant(targets)
	tf_predictions = tf.constant(predictions)

	xent_loss = tf.contrib.seq2seq.sequence_loss(tf_predictions, tf_targets, eval_mask,
							average_across_batch=False, average_across_timesteps=False)
	perp_loss = tf.contrib.seq2seq.sequence_loss(tf_predictions, tf_targets, eval_mask,
							average_across_batch=False, average_across_timesteps=False,
							softmax_loss_function = metrics.perplexity)

	with tf.Session() as sess:
		[xent_loss_output, perp_loss_output] = sess.run([xent_loss, perp_loss])

	print("CROSS-ENTROPY according to tf.nn.spare_softmax_cross_entropy_with_logits()")
	print(xent_loss_output)

	print("PERPLEXITY")
	print(perp_loss_output)




def bleu_and_wer_test(references, hypotheses):
	individual_bleus = metrics.bleu(references, hypotheses, average_across_batch=False)
	average_bleu = metrics.bleu(references, hypotheses)

	individual_wers = metrics.batch_word_error_rate(references, hypotheses, average_across_batch=False)
	average_wer = metrics.batch_word_error_rate(references, hypotheses)

	for i in range(len(references)):
		print("Translation {}".format(i))
		print("\t Reference: {}".format(references[i]))
		print("\tHypotheses: {}".format(hypotheses[i]))
		print("\tBLEU Score: {}".format(individual_bleus[i]))
		print("\t       WER: {}".format(individual_wers[i]))
	print()

	print("                Average BLEU score from metrics.bleu: {}".format(average_bleu))
	print("Average BLEU score computed from the individual ones: {}".format(sum(individual_bleus) / len(references)))
	print()

	print("                Average WER from metrics.bleu: {}".format(average_wer))
	print("Average WER computed from the individual ones: {}".format(sum(individual_wers) / len(references)))

references = ["i am so happy to be visiting the beach !",
			"do you really want to eat that melon ?",
			"no",
			"heck , yeah",
			"can you bleu ?",
			"i know how to seriously ramble on when I am trying to test out certain metrics",
			"what does the word apple mean to you ; do you think red , yellow , green , sweet , crisp ?"
]

hypotheses = ["are you that angry to be visiting the beach ?",
		"you do really want to eat that melon , don't you ?",
		"no , not really",
		"darn straight , yes",
		"do you know gleu ?",
		"to seriously ramble on when i am trying to test out certain metrics , i know how",
		"what kind of fruits do you like to eat ? red , yellow, green, sweet, crisp ones ?"
]

if __name__ == "__main__":
	#bleu_and_wer_test(references, hypotheses)
	perplexity_test()
