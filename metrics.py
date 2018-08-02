"""
Metrics for measuring the accuracy of a chatbot's/MT model's predictions
"""
import tensorflow as tf
import nltk
import numpy as np

def perplexity(logits = None, labels = None):
	"""
	:param tf.Tensor logits: The predicted logits of dimensions [batch_size*max_time_step, num_output_classes]
	:param tf.Tensor labels: The target labels of dimensions [batch_size*max_time_step]

	:returns: Losses of dimension [batch_size*max_time_step]
	:rtype:   tf.Tensor
	"""
	return tf.exp( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

smoothing_func = nltk.translate.bleu_score.SmoothingFunction(epsilon=0.1).method1
def bleu(references, hypotheses, average_across_batch=True):
	"""
	Compute BLEU-4

	:param list(list(str))           references: The references (i.e. target labels) where references[i][j] is the jth token of the ith reference
	:param list(list(str))           hypotheses: The hypothesized translations/responses where hypotheses[i][j] is the jth token of the ith hypothesis
	:param bool            average_across_batch: Whether to average the BLEU scores across samples

	:returns: A numpy vector of BLEU scores if not averaging, a scalar BLEU score otherwise
	:rtype: np.ndarray or float
	"""
	#Note the extra bracketes around references[i] - BLEU lets you give a list of reference translations
	losses = 100. * np.array(
		[
			nltk.translate.bleu_score.sentence_bleu([" ".join(references[i])],
								 " ".join(hypotheses[i]),
								 smoothing_function = smoothing_func)
			for i in range(len(references))
		],
		dtype=np.float32)

	if average_across_batch:
		return np.sum(losses) / len(references)
	return losses

def batch_word_error_rate(references, hypotheses, average_across_batch=True):
	"""
	:param list(list(str))           references: The correct reference sequence
	:param list(list(str))           hypotheses: The hypothesized sequence
	:param bool            average_across_batch: Whether to average the BLEU scores across samples

	:returns: A numpy vector of word error rates if not averaging, a scalar word error rate otherwise
	:rtype: np.ndarray or float
	"""
	losses = np.array([word_error_rate(references[i], hypotheses[i]) for i in range(len(references))], dtype=np.float32 )

	if average_across_batch:
		return np.sum(losses) / len(references)
	return losses

def word_error_rate(reference, hypothesis):
	"""
	:param list(str)  reference: The correct reference sequence
	:param list(str) hypothesis: The hypothesized sequence

	:returns: The word error rate
	:rtype: float
	"""

	distance = np.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=np.uint16)
	distance = distance.reshape((len(reference) + 1, len(hypothesis) + 1))
	for i in range(len(reference) + 1):
    		for j in range(len(hypothesis) + 1):
        		if i == 0:
            			distance[0][j] = j
        		elif j == 0:
            			distance[i][0] = i
	for i in range(1, len(reference) + 1):
    		for j in range(1, len(hypothesis) + 1):
        		if reference[i - 1] == hypothesis[j - 1]:
            			distance[i][j] = distance[i - 1][j - 1]
        		else:
            			substitution = distance[i - 1][j - 1] + 1
            			insertion = distance[i][j - 1] + 1
            			deletion = distance[i - 1][j] + 1
            			distance[i][j] = min(substitution, insertion, deletion)



	return float(distance[len(reference)][len(hypothesis)]) / len(reference)
