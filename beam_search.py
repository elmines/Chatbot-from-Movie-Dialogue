import tensorflow as tf
import collections


class DiverseBeamSearchDecoderOutput(collections.namedtuple("DiverseBeamSearchDecoderOutput",
                           ("scores", #output of an RNN cell
			    "predicted_ids" #The IDS just predicted
			    "parent_ids"
			   )
							   )
											):

	"""
	Args:
		scores - output of an RNN cell (Tensor of dimensions [batch_size, num_groups, group_size, ...]
		predicted_ids - The IDs just predicted (Tensor of dimensions [batch_size, num_groups, group_size])
		parent_ids - The IDs already predicted (Tensor of dimensions [batch_size, num_groups, group_size, num_time_steps])
	"""
		
	pass

class DiverseBeamSearchDecoderState(
	collections.namedtuple("DiverseBeamSearchDecoderState",
                           ("cell_state", "log_probs", "augmented_probs", "finished", "lengths"))):

	"""
	cell_state - The state of the decoder cell (Tensor of dimensions [batch_size, num_groups, group_size, ...]
	log_probs - The log probabilities of the generated sequences (Tensor of dimensions [batch_size, num_groups, group_size]
	augumented_probs - The log probabilities of the generated sequences, plus the diversity term (Tensor of dimensions [batch_size, num_groups, group_size]
	finished - The state of all the beams as being finished or not (boolean Tensor of dimensions [batch_size, num_groups, group_size]
	"""
	pass


def cos_sim(word_a, word_b):
	pass



class DiverseBeamSearchDecoder(tf.contrib.seq2seq.Decoder):
	"""
	Allows the addition of an additional term to the beam search optimization equation
        """

	def __init__(cell,
       			embedding,
                        start_token,
                        end_token,
                        initial_state,
			batch_size,
                        beam_width,
			num_groups,
			word_sim = cos_sim,
			sentence_sim = None,
                        output_layer = None,
			strength = 0.0 #Weight to be applied to the additional term
                        ):

		"""
		cell - An instance of RNN cell
		embedding - An embedding matrix for decoded words
		start_token - Initial scalar token to feed into decoder for each sample
        	end_token - Scalar token that causes the decoder to stop for a beam
		initial_state - Initial state of the decoder cell
		batch_size - The number of samples in the batch
		beam_width - Number of beams to maintain in decoding
		num_groups - Number of diverse groups to maintain
		word_sim - Function taking two word ids and returning a scalar describing their similarity (cannot specify both word_sim or sentence_sim)
		sentence_sim - Function taking two sequences of word ids and returning a scalar describing their similarity
        	output_layer - A layer (not a function) to apply to RNN outputs
        	"""
		self._cell = cell
		self._embedding = embedding

		self._start_token = 1d_tensor(start_token)
		self._end_token = 1d_tensor(end_token)

		self._initial_cell_state = initial_state

		self._beam_width = beam_width
		self._num_groups = num_groups
		if self._beam_width % self._num_groups != 0:
			raise ValueError("beam_width must be a multiple of num_groups")
		self._group_size = self._beam_width // self._num_groups

		if word_sim and sentence_sim:
			raise ValueError("Cannot specify both word_sim and sentence_sim")
		

		if output_layer and not instanceof(output_layer, tf.layers.Layers):
			raise TypeError("At least for now, output_layer must be a Layer object; received object of type %s" % type(output_layer))
		self._output_layer = output_layer
		



		self._search_style = search_style

		self._batch_size = batch_size #FIXME: Have the user pass this in as a parameter, or compute it from another input?

                self._finished_beams = tf.zeros([self.batch_size, self.beam_width], dtype=tf.bool)

		#The "zeros" are, expectedly, converted to `False` values
		self._finished = tf.zeros(self.batch_size, dtype=tf.bool)

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def tracks_own_finished_state():
		"""
		Like tf.contrib.seq2seq.BeamSearchDecoder, our decoder's shuffling of beams will confuse tf.contrib.seq2seq.dynamic_decode
		Returns `True`
		"""
		return True

	@property
	def output_size(self):
		"""
		Based on implementation in tf.contrib.seq2seq.BeamSearchDecoder
		"""
		return DiverseBeamSearchDecoderOutput(
							scores = tf.TensorShape([self._num_groups, self._group_size]),
							predicted_ids = tf.TensorShape([self._num_groups, self._group_size]),
							parent_ids = tf.TensorShape([self._num_groups, self._group_size])    #FIXME: Why didn't the original BeamSearchDecoder include something like max_time_step (or None)?
						)
	@property
	def output_dtype(self):
		"""
		Based on implementation in tf.contrib.seq2seq.BeamSearchDecoder
		"""
		#We do make somewhat of an assumption here, as in fact the output type could be different after applying the output layer
		dtype = tf.contrib.framework.nest.flatten(self._initial_cell_state)[0].dtype 

		return DiverseBeamSearchDecoderOutput(scores = tf.contrib.framework.nest.map_structure(lambda _: dtype, self._rnn_output_size())
							predicted_ids = tf.int32, parent_ids = tf.int32
						)

	def initialize(self, name=None):
		"""
		Returns
			finished - A tf.bool array of size [batch_size] denoting which samples are finished (obviously all `False` initially)
			initial_inputs - The initial inputs to be fed into the decoder cell (Tensor of dimensions [batch_size, num_groups, group_size])
			initial_state - The state with which to initialize the decoder cell	
		"""
		with tf.name_scope(name):
			#FIXME: Find another way to pass along the name without wasting memory using tf.identity? 
			finished = tf.identity(self._finished)

			initial_inputs = tf.fill([self._batch_size, self._num_groups, self._group_size], self._start_token)

			initial_cell_state = tf.reshape( tf.tile_batch(self._initial_cell_state, beam_width), [self._batch_size, self._num_groups, self._group_size, -1] )

			initial_augmented_probs = tf.fill([self._batch_size, self._num_groups, self._group_size], -np.Inf)
			initial_finished = self._finished_beams
			initial_lengths = tf.zeros([self._batch_size, self._num_groups, self._group_size])

			initial_state = DiverseBeamSearchDecoderState( cell_state=initial_cell_state,
									augmented_probs = initial_augmented_probs,
									finished = initial_finished,
									lengths = initial_lengths
								)

		return (finished, initial_inputs, initial_state)

	def step(self, time, inputs, state):
		"""
		time - a scalar indicating the current time step
		inputs - Tensor of dimensions [batch_size, num_groups, group_size]
		state - Tensor (or someother collection?) of dimensions [batch_size, num_groups, group_size] where each element is a DiverseBeamDecoderState object
		"""

		batch_size = self._batch_size
		num_groups = self._num_groups
		group_size = self._group_size
		end_token = self._end_token

		#Tensor of shape [batch_size, num_groups, group_size, ...]
		#Access cell_state from the named tuple
		state = state.cell_state 

		logits = tf.contrib.framework.nest.map_structure(lambda cell_inputs, cell_state: self._cell(cell_inputs, cell_state), inputs, state)

		#logits - Tensor of dimensiosn [batch_size, num_groups, group_size, num_output_classes]



		outputs = None
		next_state = None
		next_input = None
		finished = None

		return (outputs, next_state, next_input, finished)

		pass

	def finalize(self, outputs, final_state, sequence_lengths, name=None):
		"""
		outputs - A Tensor of dimensions [batch_size, num_groups, group_size, max_time_step]
		final_state - A Tensor of dimensions [batch_size, num_groups, group_size]
		sequence_lengths - A Tensor of dimensions [batch_size, num_groups, group_size]
		"""
		pass

		
	

	def _rnn_output_size(self):
		"""
		Based on implementation in tf.contrib.seq2seq.BeamSearchDecoder
		"""
		size = self._cell.output_size
		if not self._output_layer: return size

		output_shape_with_unknown_batch = tf.contrib.framework.nest.map_structure(lambda s: tf.TensorShape([None]).concatenate(s), size)

		#Apparently we need that extra dimension for the sole purpose of calling compute_output_shape . . .
		layer_output_shape = self._output_layer.compute_output_shape(output_shapw_with_unknown_batch)

		#. . . since we just discard it right here
		return tf.contrib.framework.nest.map_structure(lambda s: s[1:], layer_output_shape)


	def _beam_scores(self, beams):
		"""
		beams - Tensor of dimensions [batch_size, num_groups, group_size, latest_time_step]

		Returns
			scores - Tensor of dimensions [batch_size, num_groups, group_size]
		"""

		return None


	def top_k(self, groups):
		"""
		groups - tensor of dimensions [batch_size, num_groups, group_size, latest_time_step]
		"""

		

