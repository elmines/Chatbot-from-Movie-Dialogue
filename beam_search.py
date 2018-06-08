import tensorflow as tf


class CustomBeamSearchDecoder: #(tf.Decoder)
	"""
	Allows the addition of an additional term to the beam search optimization equation
        """

	def __init__(cell,
       			embedding,
                        start_token,
                        end_token,
                        initial_state,
                        beam_width,
                        output_fn,
			strength = 0.0 #Weight to be applied to the additional term
                        search_style=None):

		"""
		cell - An instance of RNN cell
		embedding - An embedding matrix for decoded words
		start_token - Initial token to feed into decoder for each sample
        	end_token - Token that causes the decoder to stop for a beam
		initial_state - Initial state of the decoder cell
		beam_width - Number of beams to maintain in decoding
        	output_layer - A function (need not be a layer) to apply to RNN outputs
        	search_style - FUTURE WORK: style of beam search to use (diverse, affectively diverse, etc.)
        	"""
		self.cell = cell
		self.embedding = embedding
		self.start_tokens = start_tokens
		self.beam_width = beam_width
		self.output_layer = output_layer
		self.search_style = search_style

		self._batch_size = 256 #FIXME: Have the user pass this in as a parameter, or compute it from another input?

                self.finished_beams = tf.Variable(0, tf.int32)

	def decode():
		"""
		Returns
			outputs - A tensor of dimensions [batch_size, max_sequence_length, output_size]
                        true_sequence_lengths - A tensor of dimensions [batch_size] describing the true lengths of the generated sequences
		"""

		return _, _

		
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
	

	def finalize(self, outputs, final_state, sequence_lengths, name=None):
		pass

	def initialize(self, name=None):
		pass

	def step(self, time, inputs, state):
		pass

