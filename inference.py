import tensorflow as tf
import time
import sys

#TODO: Move functions like batch_feeds to a module separate from training and inference
from training import batch_feeds 
from training import merge_dicts

def beam_frame(beams):
	"""
	:param list(list(str)) beams: Set of beams for each prompt

	:returns Dataframe of the beams with columns \"beams_0\", \"beams_1\", . . . \"beams_{beam_width - 1}\"
	:rtype pd.DataFrame
	"""
	out_frame = pd.DataFrame()
	beam_width = len(beams[0])
	for i in range(beam_width):
		beam_col_i = [beam_set[i] for beam_set in beams]
		out_frame["beams_{}".format(i)] = beam_col_i
	return out_frame

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

		actual_batch_size = len( feed_dict[data_placeholders.input_data] )
		source_lengths = feed_dict[data_placeholders.source_lengths]

		augmented_feed_dict = merge_dicts(feed_dict, feeds)
		fetch_outputs = sess.run(fetches, augmented_feed_dict)
		fetches_list.extend(fetch_outputs)

		sys.stderr.write("Test batch {}, {} seconds\n".format(batch_i, time.time() - start_time))


	return fetches_list

def show_response(prompt_int, beams, prompts_int_to_vocab, answers_int_to_vocab, pad_q, pad_a, answer_int = None):
	"""
	Display the model's response

	:param prompt_int: A 1-D iterable of integers
	:param beams: Response beams as a 2-D iterable or a single response as a 1-D iterable
	"""
	prompt_text = [prompts_int_to_vocab[tok] for tok in prompt_int if tok != pad_q]
	print("Prompt")
	print("  Word Ids: {}".format([i for i in prompt_int if i != pad_q]))
	print("      Text: {}".format(prompt_text))
    
	if answer_int is not None:
		answer_text = [answers_int_to_vocab[tok] for tok in answer_int if tok != pad_a]
		print("Target Answer")
		print("  Word Ids: {}".format([i for i in answer_int if i != pad_a]))
		print("      Text: {}".format(answer_text))

	try:
		beams[0][0]
	except:
		beams = [beams] #If only passed in one beam as a vector, add an extra dimension
	beam_width = len(beams[0])

	for i in range(beam_width):
		beam = beams[:, i]
		print()
		if i == 0:
			print("Best prediction")
		else:
			print("Prediction")

		pred_text = [answers_int_to_vocab[tok] for tok in beam if tok != pad_a]
		print('  Word Ids: {}'.format([i for i in beam if i != pad_a]))
		print('      Text: {}'.format(pred_text))

