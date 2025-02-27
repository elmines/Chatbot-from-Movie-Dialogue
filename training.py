"""
Module for training a :py:class:`models.Seq2Seq` model
"""
import tensorflow as tf
import numpy as np
import sys
import time

def merge_dicts(*dicts):
	"""
	Merges two dictionaries with disjoint sets of keys

	Becasuse it checks for identical keys, this function runs in time quadratic with the number of keys and was never meant for scalability.

	:param list(dict) dicts: One or more dictionaries to merge

	:raises ValueError: if a key is found in more than one dictionary

	:returns: The combined dictionary
	:rtype: dict
	"""
	merged = {}
	for dictionary in dicts:
		for key in dictionary:
			if key in merged.keys():
				raise ValueError("{} is in more than one dictionary".format(key))
			merged[key] = dictionary[key]
	return merged


############FEEDDING DATA################
def parallel_shuffle(source_sequences, target_sequences):
	"""
	:param list source_sequences: A list of input sequences
	:param list target_sequences: A list of parallel target sequences

	:raises ValueError: if there are different numbers of source and target sequences

	:returns: The shuffled sequences (without affecting the originals)
	:rtype: tuple(list,list)
	"""

	if len(source_sequences) != len(target_sequences):
		raise ValueError("Cannot shuffle parallel sets with different numbers of sequences")
	indices = np.random.permutation(len(source_sequences))
	shuffled_source = [source_sequences[indices[i]] for i in indices]
	shuffled_target = [target_sequences[indices[i]] for i in indices]

	return (shuffled_source, shuffled_target)
def pad_sentence_batch(sentence_batch, pad_token):
	"""
	Pads sequences to equal lengths

	:param list(list) sentence_batch: The sequences to be padded
	:param                 pad_token: The token with which to pad the sequences

	:returns: The padded sequences
	:rtype: list(list)
	"""

	max_sentence_length = max([len(sentence) for sentence in sentence_batch])
	return [sentence + [pad_token] * (max_sentence_length - len(sentence)) for sentence in sentence_batch]

def single_batch(data_placeholders, questions_batch, answers_batch, pad_token):
	"""
	Maps TensorFlow placeholder variables to a batch of training data

	:param tf_collections.DataPlaceholders   data_placeholders: The placeholders for which to feed data
	:param list(list(int))                     questions_batch: A batch of prompts
	:param list(list(int))                       answers_batch: A batch of responses
	:param int                                       pad_token: The token with which to pad sequences to equal lengths

	:returns: A feed dictionary mapping the placeholders in data_placeholders to the data for one batch
	:rtype: dict(tf.Tensor,object)
	"""
	source_lengths = np.array( [len(sentence) for sentence in questions_batch] )
	target_lengths = np.array( [len(sentence) for sentence in answers_batch])

	pad_prompts_batch = np.array(pad_sentence_batch(questions_batch, pad_token))
	pad_answers_batch = np.array(pad_sentence_batch(answers_batch, pad_token))

	#DataPlaceholder variables
	feed_dict = {
			data_placeholders.input_data     : pad_prompts_batch,
			data_placeholders.targets        : pad_answers_batch,
			data_placeholders.source_lengths : source_lengths,
			data_placeholders.target_lengths : target_lengths
		}
	return feed_dict

def batch_feeds(data_placeholders, questions_int, answers_int, batch_size, pad_token):
	"""
	Batches training data and returns a mapping from TensorFlow placeholder variables to the data

	:param tf_collections.DataPlaceholders data_placeholders: The placeholders for which to feed data
	:param list(list(int))                     questions_int: The prompts to be batched
	:param list(list(int))                       answers_int: The corresponding responses to be batched
	:param int                                    batch_size: The minibatch size
	:param int                                     pad_token: The token with which to pad sequences to equal lengths

	:returns: An iterator of feed dictionaries mapping the placheholders in data_placeholders to the data for one batch
	:rtype: Iterator(dict(tf.Tensor,object))
	"""
	for batch_i in range(0, len(questions_int)//batch_size):
		start_i = batch_i * batch_size
		questions_batch = questions_int[start_i:start_i + batch_size]
		answers_batch = answers_int[start_i:start_i + batch_size]
		yield single_batch(data_placeholders, questions_batch, answers_batch, pad_token)

	if batch_size > 1:
		remainder = len(questions_int) % batch_size
		questions_batch = questions_int[-remainder:]
		answers_batch = questions_int[-remainder:]
		yield single_batch(data_placeholders, questions_batch, answers_batch, pad_token)
	
#TODO: Add support for logging
class Trainer(object):
	"""
	Class maintaining the state of training for a :py:class:`models.Seq2Seq` model
	"""
	def __init__(self, save_fn, epochs_completed=0, max_epochs=50, best_valid_cost = float("inf"), stalled_steps = 0, max_stalled_steps=float("inf")):
		"""
		:param callable            save_fn: A function with signature save_fn(tf.Session)
		:param int        epochs_completed: Epochs of training already completed
		:param int              max_epochs: Total epochs to complete
		:param float       best_valid_cost: Best validation loss observed thus far
		:param int           stalled_steps: The number of consecutive stalled validation steps
		:param int       max_stalled_steps: The maximum number of stalled steps before stopping training early
		"""

		if not callable(save_fn):
			raise ValueError("save_fn must be callable.")
		self._save_fn  = save_fn
		self._epochs_completed = epochs_completed
		self._max_epochs = max_epochs
		self._stalled_steps = stalled_steps
		self._max_stalled_steps = max_stalled_steps
		self._record = best_valid_cost

	def save_latest(self, sess):
		"""
		Saves the current state of the model and prints the number of epochs completed

		:param tf.Session sess: The TensorFlow session in which to save the model
		"""
		self.save_fn(sess)
		print("{}/{} epochs completed".format(self.epochs_completed, self.max_epochs))

	def check_validation_loss(self, sess, validation_cost):
		"""
		Checks the validation loss against the current record, and saves the model if a new record has been reached

		:param tf.Session            sess: The TensorFlow session in which to save the model
		:param float      validation_cost: The latest validation loss
		"""
		if not( validation_cost >= self._record ):
			self._stalled_steps = 0
			self._record = validation_cost
			print("New record for validation cost!")
		#	self.save_fn(self._best_model_path, sess)
		else:
			self._stalled_steps += 1
			print("No new record for validation cost--stalled for {}/{} steps".format(self._stalled_steps, self._max_stalled_steps))

	def inc_epochs_completed(self):
		"""
		Increments the number of epochs completed
		"""
		self._epochs_completed += 1

	@property
	def save_fn(self):
		"""
		callable: The function used to save the current model
		"""
		return self._save_fn

	@property
	def finished(self):
		"""
		bool: Whether training is completed (based on the number of epochs completed and the number of stalled steps)
		"""
		return (self.epochs_completed >= self.max_epochs) or (self.stalled_steps >= self.max_stalled_steps)

	@property
	def epochs_completed(self):
		"""
		int: The number of epochs of training completed
		"""
		return self._epochs_completed

	@property
	def max_epochs(self):
		"""
		int: The maximum number of epochs for which to train
		"""
		return self._max_epochs

	@property
	def stalled_steps(self):
		"""
		int: The current number of consecutive stalled steps
		"""
		return self._stalled_steps

	@property
	def max_stalled_steps(self):
		"""
		int: The maximum number of stalled steps to tolerate before ending training
		"""
		return self._max_stalled_steps

	@property
	def best_valid_cost(self):
		"""
		float: The best validation loss recorded thus far
		"""
		return self._record
		

def training_loop(sess, model, trainer, data, train_feeds=None, valid_feeds=None, train_batch_size=64, valid_batch_size=64, min_epochs_before_validation=2):
	"""
	Trains a :py:class:`models.Seq2Seq` model

	:param tf.Session                                      sess: The TensorFlow session with which to run model's operations
	:param models.Seq2Seq                                 model: The model representing the computations to be performed
	:param Trainer                                      trainer: The Trainer, which controls the flow of training
	:param data.Data                                       data: Data object with members **train_prompts**, **train_answers**, **valid_prompts**, **valid_answers**
	:param dict(tf.Tensor,object)                   train_feeds: Feed-dict when training
	:param dict(tf.Tensor,object)                   valid_feeds: Feed-dict when running validation
	:param int                                 train_batch_size: Minibatch size during training
	:param int                                 valid_batch_size: Minibatch size during validation
	:param int                     min_epochs_before_validation: Minimum number of epochs to perform before doing validation as well

	"""

	data_placeholders = model.data_placeholders

	#Data
	(train_prompts_int, train_answers_int) = (data.train_prompts.indices, data.train_answers.indices)
	(valid_prompts_int, valid_answers_int) = (data.valid_prompts.indices, data.valid_answers.indices)

	#We could rewrite the padding functions to just pad with zeros, but parameterizing them
	# with pad_int allows for future change
	pad_int = 0 
	
	#Logging
	num_tokens = lambda feed_dict: sum(feed_dict[data_placeholders.target_lengths])

	#Fetches
	(train_op, train_cost, valid_cost) = (model.train_op, model.train_cost, model.valid_cost)

	display_step = 100
	while not trainer.finished:
		print("Shuffling training data . . .")
		(train_prompts_int, train_answers_int) = parallel_shuffle(train_prompts_int, train_answers_int)

		valid_check_no = 1
		tot_train_tokens = 0
		tot_train_loss = 0
		train_start_time = time.time()
		for batch_i, feed_dict in \
			enumerate(batch_feeds(data_placeholders, train_prompts_int, train_answers_int, train_batch_size, pad_int)):
    
			augmented_feed_dict = merge_dicts(feed_dict, train_feeds)
			_, loss = sess.run([train_op, train_cost], augmented_feed_dict)

			batch_tokens = num_tokens(feed_dict)
			tot_train_tokens += batch_tokens
			tot_train_loss += batch_tokens*loss
    
			if batch_i % display_step == 0:
				duration = time.time() - train_start_time
				avg_train_loss = tot_train_loss / tot_train_tokens
				print('Epoch {:>3}/{} Batch {:>4}/{} - Loss-per-Token: {:>9.6f}, Seconds: {:>4.2f}'
              				.format(trainer.epochs_completed+1, trainer.max_epochs, batch_i, len(train_prompts_int) // train_batch_size, avg_train_loss, duration),
                 			flush=True)
				tot_train_tokens = 0
				tot_train_loss = 0
				train_start_time = time.time()
		trainer.inc_epochs_completed()
		trainer.save_latest(sess)
		#TODO: For simplicity's sake we're just performing validation after each epoch
		#VALIDATION CHECK
		if trainer.epochs_completed >= min_epochs_before_validation:
			print("Shuffling validation data . . .")
			(valid_prompts_int, valid_answers_int) = parallel_shuffle(valid_prompts_int, valid_answers_int)
			tot_valid_tokens = 0
			tot_valid_loss = 0
			valid_start_time = time.time()
			for batch_ii, feed_dict in enumerate(batch_feeds(data_placeholders, valid_prompts_int, valid_answers_int, valid_batch_size, pad_int)):
				augmented_feed_dict = merge_dicts(feed_dict, valid_feeds)
				[loss] = sess.run([valid_cost],augmented_feed_dict)
				batch_tokens = num_tokens(feed_dict)
				tot_valid_tokens += batch_tokens
				tot_valid_loss += batch_tokens*loss
			duration = time.time() - valid_start_time
			avg_valid_loss = tot_valid_loss / tot_valid_tokens
			print("Processed validation set in {:>4.2f} seconds".format(duration))
			print("Loss-per-Token = {}".format(avg_valid_loss))
			trainer.check_validation_loss(sess, avg_valid_loss)
			valid_check_no += 1
