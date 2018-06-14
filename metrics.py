import tensorflow as tf
import nltk
import numpy as np

def sequence_perplexity(labels = None, logits = None):
	"""
	`labels` - the ground truth target values: Tensor of dimensinos [batch_size*max_time_step]
	`logits` - the predictions of dimensions [batch_size*max_time_step, num_output_classes]
	Returns
		losses - the loss of dimension [batch_size*max_time_step]
	"""
	 return tf.exp( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) )
	
def bleu(references, hypotheses, average_across_batch=True):
	"""
	Compute BLEU-4

	`references` - a list of strings, with one reference per hypothesis
	`hypotheses` - a list of strings
	"""
	losses = np.array([nltk.translate.sentence_bleu([targets[i]], predictions[i]) for i in range(len(targets))],
			dtype=np.float32)

	if average_across_batch:
		return np.sum(losses)
	return losses

def batch_word_error_rate(references, hypthoses, average_across_batch=True):
	"""
	`references` - list of strings where each string is a reference translation
	`hypotheses` - list of strings where each string is a hypothesis
	"""
	losses = np.array([word_error_rate(references[i].split(" "), hypotheses[i].split(" ")) for i in range(len(references))], dtype=np.float32 )

	if average_across_batch:
		return np.sum(losses)
	return losses

def word_error_rate(reference, hypothesis):
	"""
	reference - list of strings where each string is a token
	hypothesis - list of strings where each string is a token

	Returns word error rate(insert, delete or substitution).
	"""
		
	distance = numpy.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=numpy.uint16)
	distance = distance.reshape((len(reference) + 1, len(hypothesis) + 1))
	for i in range(len(reference) + 1):
    		for j in range(len(hypothesis) + 1):
        		if i == 0:
            			distance[0][j] = j
        		elif j == 0:
            			distance[i][0] 	
	for i in range(1, len(reference) + 1):
    		for j in range(1, len(hypothesis) + 1):
        		if reference[i - 1] == hypothesis[j - 1]:
            			distance[i][j] = distance[i - 1][j - 1]
        		else:
            			substitution = distance[i - 1][j - 1] + 1
            			insertion = distance[i][j - 1] + 1
            			deletion = distance[i - 1][j] + 1
            			distance[i][j] = min(substitution, insertion, deletion)
	result = float(distance[len(reference)][len(hypothesis)]) / len(reference) * 100
