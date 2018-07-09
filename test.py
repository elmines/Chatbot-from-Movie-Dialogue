import tensorflow as tf
import time
import sys

#TODO: Move functions like batch_feeds to a module separate from training and query
from training import batch_feeds 
from training import merge_dicts

def infer(sess, model, prompts_int, feeds, fetches, pad_int, batch_size=64):
	"""
	:param tf.Session                    sess: The active TensorFlow session
	:param models.Seq2Seq               model: The dialog generation model
	:param list(list(int))            prompts: The prompts for which to generate replies
	:param dict(tf.Tensor, tf.Tensor)   feeds: Additional feeds for sess.run (like dropout probability)
	:param list(tf.Tensor)            fetches: Outputs to fetch from sess.run

	:returns List of outputs for each samples
	:rtype list(type(fetches))

	"""

	data_placeholders = model.data_placeholders 
	fetches_list = []


	#FIXME: Don't pass in prompts twice--it works, but it's a bad practice
	for batch_i, feed_dict in enumerate( batch_feeds(data_placeholders, prompts_int, prompts_int, batch_size, pad_int) ):
		start_time = time.time()

		augmented_feed_dict = merge_dicts(feed_dict, feeds)
		fetch_outputs = sess.run(fetches, augmented_feed_dict)
		fetches_list.extend(fetch_outputs)

		sys.stderr.write("Test batch {}, {} seconds\n".format(batch_i, time.time() - start_time))

		break

	return fetches_list
