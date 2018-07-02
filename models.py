import tensorflow as tf
import collections

#Local modules
import tf_collections
import loss_functions
import metrics

def create_placeholders():
	#                                          batch_size  time
	input_data =     tf.placeholder(tf.int32, [None,       None], name='input_data')
	targets =        tf.placeholder(tf.int32, [None,       None], name='targets')
	source_lengths = tf.placeholder(tf.int32, [None],             name="source_lengths")
	target_lengths = tf.placeholder(tf.int32, [None],             name="target_lengths")

	placeholders = tf_collections.DataPlaceholders(input_data=input_data, targets=targets,
					source_lengths=source_lengths, target_lengths=target_lengths)

	return placeholders


def _dropout_cell(rnn_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        return tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
def _multi_dropout_cell(rnn_size, keep_prob, num_layers):    
        return tf.contrib.rnn.MultiRNNCell( [_dropout_cell(rnn_size, keep_prob) for _ in range(num_layers)] )
def _process_decoding_input(target_data, go_token):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        batch_size = tf.shape(target_data)[0]
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat( [tf.fill([batch_size, 1], go_token), ending], 1)
        return dec_input
def _decoding_layer(enc_state, enc_outputs, dec_embed_input, dec_embeddings, dec_cell, attn_size, output_layer, source_lengths, target_lengths, go_token, eos_token):
	batch_size = tf.shape(source_lengths)[0]
	
	attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=attn_size, memory=enc_outputs, memory_sequence_length=source_lengths)
	attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, attention_layer_size=dec_cell.output_size)
	
	init_attn_dec_state = attn_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_state)
	
	decoder_gen = lambda helper: tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, init_attn_dec_state,
		output_layer = output_layer)
	
	with tf.variable_scope("decoding") as scope:
		#TRAINING
		train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_lengths)
		train_decoder = decoder_gen(train_helper)
		train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, scope=scope)
		train_logits = train_outputs.rnn_output
	
		#scope.reuse_variables()
		#INFERENCE
		infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens = tf.tile([go_token], [batch_size]), end_token = eos_token)
		infer_decoder = decoder_gen(infer_helper)
		infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, impute_finished=True, maximum_iterations=tf.round(tf.reduce_max(source_lengths) * 2), scope=scope)
		infer_ids = infer_outputs.sample_id
	
	return train_logits, infer_ids


class Seq2Seq(object):

	def __init__(self, placeholders, enc_embeddings, dec_embeddings, go_token, eos_token,
			num_layers=1, rnn_size=1024, attn_size=256, output_layer=None,
			learning_rate=0.0001):
		"""
		placeholders - A DataPlaceholders namedtuple
		learning_rate - A constant (you can't decay it over time)
		"""

		self._data_placeholders = placeholders
		self._input_data = placeholders.input_data
		self._targets = placeholders.targets
		self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	
	        #Both of these could be necessary for subclasses with custom loss functions
		self._source_lengths = placeholders.source_lengths
		self._target_lengths = placeholders.target_lengths
		self._enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, self._input_data)
		self._dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, _process_decoding_input(self._targets, go_token))
	
	
		forward_cell = _multi_dropout_cell(rnn_size, self._keep_prob, num_layers)
		backward_cell = _multi_dropout_cell(rnn_size, self._keep_prob, num_layers)
		enc_outputs, enc_states =  tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_cell, cell_bw = backward_cell,
										sequence_length = self._source_lengths,
										inputs = self._enc_embed_input, dtype=tf.float32)
		concatenated_enc_output = tf.concat(enc_outputs, -1)
		init_dec_state = enc_states[0] 
	
	
		dec_cell = _multi_dropout_cell(rnn_size, self._keep_prob, num_layers)
		self._train_logits, self._infer_ids = _decoding_layer(init_dec_state, concatenated_enc_output, self._dec_embed_input, dec_embeddings,
	                        dec_cell, attn_size, output_layer, self._source_lengths, self._target_lengths, go_token, eos_token)
	
		self._eval_mask = tf.sequence_mask(self._target_lengths, dtype=tf.float32)
		self._xent = tf.contrib.seq2seq.sequence_loss(self._train_logits, self._targets, self.eval_mask)
		self._perplexity = tf.contrib.seq2seq.sequence_loss(self._train_logits, self._targets, self.eval_mask, softmax_loss_function=metrics.perplexity)
	
		self._optimizer = tf.train.AdamOptimizer(learning_rate)


	#####COST/LOSS######

	@property
	def train_op(self):
		raise NotImplementedError("Abstract method")

	@property
	def train_cost(self):
		raise NotImplementedError("Abstract method")

	@property
	def valid_cost(self):
		raise NotImplementedError("Abstract method")

	@property
	def xent(self):
		return self._xent

	@property
	def perplexity(self):
		return self._perplexity

	@property
	def eval_mask(self):
		return self._eval_mask

	@property
	def keep_prob(self):
		return self._keep_prob



	@property
	def data_placeholders(self):
		return self._data_placeholders

	@property
	def targets(self):
		return self._targets

	@property
	def source_lengths(self):
		return self._source_lengths

	@property
	def target_lengths(self):
		return self._target_lengths

	@property
	def train_logits(self):
		return self._train_logits

	@property
	def infer_ids(self):
		return self._infer_ids

	@property
	def optimizer(self):
	        return self._optimizer
	
	@property
	def enc_embed_input(self):
		return self._enc_embed_input
	@property
	def dec_embed_input(self):
		return self._dec_embed_input

class Aff2Vec(Seq2Seq):
	def __init__(self, *args, **kwargs):
		super(Aff2Vec, self).__init__(*(args), **(kwargs))
	
		gradients = self.optimizer.compute_gradients(self.xent)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		self._train_op = self.optimizer.apply_gradients(capped_gradients)
	@property
	def train_cost(self):
		return self.xent

	@property
	def valid_cost(self):
		return self.perplexity

	@property
	def train_op(self):
		return self._train_op

class VADAppended(Seq2Seq):

	def __init__(self, placeholders, full_embeddings, go_token, eos_token,
                num_layers=1, rnn_size=1024, attn_size=256, output_layer=None,
		keep_prob = 1, learning_rate=0.0001,
 		affect_strength=0.5):
		"""
		affect_strength - hyperparameter in the range [0.0, 1.0)
		"""
		
		Seq2Seq.__init__(self, placeholders, full_embeddings, full_embeddings,go_token, eos_token,
				num_layers=num_layers,rnn_size=rnn_size,attn_size=attn_size,output_layer=output_layer, learning_rate=learning_rate)

		emot_embeddings = full_embeddings[:, -3: ]
		neutral_vector = tf.constant([5.0, 1.0, 5.0], dtype=tf.float32)
		affective_loss = loss_functions.max_affective_content(affect_strength, self.train_logits, self.targets, emot_embeddings, neutral_vector, self.eval_mask)
		self._train_affect = tf.placeholder_with_default(False, shape=())
		self._train_cost = tf.cond(self._train_affect, true_fn= lambda: affective_loss, false_fn= lambda: self.xent)

		gradients = self.optimizer.compute_gradients(self._train_cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		self._train_op = self.optimizer.apply_gradients(capped_gradients)

	@property
	def train_affect(self):
	        return self._train_affect
	
	@property
	def train_cost(self):
		return self._train_cost

	@property
	def valid_cost(self):
		return self.perplexity
	
	@property
	def train_op(self):
		return self._train_op
