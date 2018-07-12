import tensorflow as tf
import numpy as np
import sys
import time

#This function runs in quadratic time and was not meant for scalability
def merge_dicts(*dicts):
	merged = {}
	for dictionary in dicts:
		for key in dictionary:
			if key in merged.keys():
				raise ValueError("{} is in more than one dictionary".format(key))
			merged[key] = dictionary[key]
	return merged


############FEEDDING DATA################
def parallel_shuffle(source_sequences, target_sequences):
    if len(source_sequences) != len(target_sequences):
        raise ValueError("Cannot shuffle parallel sets with different numbers of sequences")
    indices = np.random.permutation(len(source_sequences))
    shuffled_source = [source_sequences[indices[i]] for i in indices]
    shuffled_target = [target_sequences[indices[i]] for i in indices]
    
    return (shuffled_source, shuffled_target)
def pad_sentence_batch(sentence_batch, pad_token):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence_length = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_token] * (max_sentence_length - len(sentence)) for sentence in sentence_batch]

def single_batch(data_placeholders, questions_batch, answers_batch, pad_token):
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
	data_placeholders - a tf_collections.DataPlaceholders object

	Returns
		a feed dictionary with mapping data_placeholders to a batch
	"""
	for batch_i in range(0, len(questions_int)//batch_size):
		start_i = batch_i * batch_size
		questions_batch = questions_int[start_i:start_i + batch_size]
		answers_batch = answers_int[start_i:start_i + batch_size]
		yield single_batch(data_placeholders, questions_batch, answers_batch, pad_token)

	remainder = len(questions_int) % batch_size
	questions_batch = questions_int[-remainder:]
	answers_batch = questions_int[-remainder:]

	yield single_batch(data_placeholders, questions_batch, answers_batch, pad_token)
	
#TODO: Add support for logging
class Trainer(object):
	"""
	Class describing the state of a models.Seq2Seq's training
	"""
	def __init__(self, best_model_path, latest_model_path, save_fn, epochs_completed=0, max_epochs=50, best_valid_cost = float("inf"), stalled_steps = 0, max_stalled_steps=float("inf")):
		"""
		:param            best_model_path: Path at which to save the model with the best validation loss
		:param          latest_model_path: Path at which to save the latest model
		:param callable           save_fn: A function with signature save_fn(path_prefix, tf.Session) -> actual_path
		:param int       epochs_completed: Epochs of training already completed
		:param int             max_epochs: Total epochs to complete
		:param float      best_valid_cost: Best validation loss observed thus far
		:param int          stalled_steps: The number of consecutive stalled validation steps
		:param int      max_stalled_steps: The maximum number of stalled steps before stopping training early
		"""

		if not callable(save_fn):
			raise ValueError("save_fn must be callable.")
		self._save_fn  = save_fn
		self._best_model_path = best_model_path
		self._latest_model_path = latest_model_path
		self._epochs_completed = epochs_completed
		self._max_epochs = max_epochs
		self._stalled_steps = stalled_steps
		self._max_stalled_steps = max_stalled_steps
		self._record = best_valid_cost


		
	def save_latest(self, sess):
		self.save_fn(self._latest_model_path, sess)
		print("{}/{} epochs completed".format(self.epochs_completed, self.max_epochs))

	def check_validation_loss(self, sess, validation_cost):
		if not( validation_cost >= self._record ):
			self._stalled_steps = 0
			self._record = validation_cost
			print("New record for validation cost!")
			self.save_fn(self._best_model_path, sess)
		else:
			self._stalled_steps += 1
			print("No new record for validation cost--stalled for {}/{} steps".format(self._stalled_steps, self._max_stalled_steps))

	def inc_epochs_completed(self):
		self._epochs_completed += 1

	@property
	def save_fn(self):
		return self._save_fn

	@property
	def finished(self):
		return (self.epochs_completed >= self.max_epochs) or (self.stalled_steps >= self.max_stalled_steps)

	@property
	def epochs_completed(self):
		return self._epochs_completed

	@property
	def max_epochs(self):
		return self._max_epochs

	@property
	def stalled_steps(self):
		return self._stalled_steps

	@property
	def max_stalled_steps(self):
		return self._max_stalled_steps

	@property
	def best_valid_cost(self):
		return self._record
		

def training_loop(sess, model, trainer, datasets, text_data, train_feeds=None, valid_feeds=None, train_batch_size=64, valid_batch_size=64, min_epochs_before_validation=2):
	"""
	data_placeholders - a tf_collections.DataPlaceholders namedtuple
	fetches - a tf_collections.Fetches namedtuple	
	train_feeds - a dictionary of extra feed-value pairs when calling sess.run for training
	valid_feeds - the validation analog of train_feeds
	Returns
		- (float) The current epoch_no
		- (float) The best validation loss computed
	"""

	data_placeholders = model.data_placeholders

	#Data
	(train_prompts_int, train_answers_int) = (datasets.train_prompts_int, datasets.train_answers_int)
	(valid_prompts_int, valid_answers_int) = (datasets.valid_prompts_int, datasets.valid_answers_int)
	(prompts_int_to_vocab, answers_int_to_vocab) = text_data.prompts_int2vocab, text_data.answers_int2vocab
	(unk_int, pad_int) = text_data.unk_int, text_data.pad_int
	
	#Logging
	num_tokens = lambda feed_dict: sum(feed_dict[data_placeholders.target_lengths])

	#Fetches
	(train_op, train_cost, valid_cost) = (model.train_op, model.train_cost, model.valid_cost)
	beams = model.beams

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

