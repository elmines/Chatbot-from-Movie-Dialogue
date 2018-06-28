import tensorflow as tf
import collections

#Local modules
import loss_functions



class Placeholders(collections.namedtuple("Placeholders", ["input_data", "targets", "source_lengths", "target_lengths", "keep_prob"])):
	pass

class Metatokens(collections.namedtuple("Metatokens", ["EOS", "GO", "PAD", "UNK"])):
	pass

#FIXME: EOS parameter
def append_eos(answers_text, answers_int):
    appended_text = [sequence + [EOS] for sequence in answers_text]
    appended_ints = [sequence + [METATOKEN_INDEX] for sequence in answers_int]
    return (appended_text, appended_ints)

def create_placeholders():
	#                                          batch_size  time
	input_data =     tf.placeholder(tf.int32, [None,       None], name='input_data')
	targets =        tf.placeholder(tf.int32, [None,       None], name='targets')
	source_lengths = tf.placeholder(tf.int32, [None],             name="source_lengths")
	target_lengths = tf.placeholder(tf.int32, [None],             name="target_lengths")

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	placeholders = Placeholders(input_data=input_data, targets=targets,
					source_lengths=source_lengths, target_lengths=target_lengths,
					keep_prob=keep_prob)

	return placeholders

class VADAppended(Seq2Seq):

	def __init__(self, placeholders, word_embeddings, num_layers, rnn_size, attn_size, output_layer, affect_strength):
		"""
		affect_strength - hyperparameter in the range [0.0, 1.0)
		"""
		
		Seq2Seq.__init__(self, placeholders, word_embeddings, word_embeddings,
					num_layers, rnn_size, attn_size, output_layer)

		emot_embeddings = word_embeddings[-3: ]

            	neutral_vector = tf.constant([5.0, 1.0, 5.0], dtype=tf.float32)
            	affective_loss = loss_functions.max_affective_content(affect_strength, self._train_logits, self.targets, emot_embeddings, neutral_vector, self.eval_mask)


		self._train_affect = tf.Variable(False, trainable=False, dtype=tf.bool)


		train_cost = tf.cond(train_affect, true_fn= lambda: affective_loss, false_fn= lambda: self.xent)
		gradients = optimizer.compute_gradients(train_cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		self.train_op = optimizer.apply_gradients(capped_gradients)

class Seq2Seq(object):

	def __init__(self, placeholders, enc_embedddings, dec_embeddings,
			num_layers, rnn_size, attn_size, output_layer,
			learning_rate):
		"""
		placeholders - A Placeholders namedtuple
		learning_rate - A constant (you can't decay it over time)
		"""

		self.input_data = placeholders.input_data
		self.targets = placeholders.targets
		self._source_lengths = placeholders.source_lengths
		self._target_lengths = placeholders.target_lengths
		self._keep_prob = placeholders.keep_prob

		batch_size = tf.shape(input_data)[0]

		self.enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, self.input_data)
		dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, process_decoding_input(targets, batch_size))


		enc_outputs, enc_states = self._encoding_layer(self.enc_embed_input, rnn_size, num_layers, keep_prob, source_lengths)    
    		concatenated_enc_output = tf.concat(enc_outputs, -1)
    		init_dec_state = enc_states[0]    
    
    
    		self._train_logits, self._infer_ids = _decoding_layer(init_dec_state, concatenated_enc_output, dec_embed_input, dec_embeddings,
                            attn_size, rnn_size, num_layers, output_layer,
                            keep_prob, source_lengths, target_lengths, batch_size
                            )

		self.eval_mask = tf.sequence_mask(target_lengths, dtype=tf.float32)
		self.xent = tf.contrib.seq2seq.sequence_loss(self._train_logits, self._targets, self.eval_mask)

		self.optimizer = tf.train.AdamOptimizer(learning_rate)

		

	def train(self):
		raise NotImplementedError("Abstract method")


	@property
	def train_logits():
		return self._train_logits

	@property
	def infer_ids():
		return self._infer_ids


	def process_decoding_input(target_data, batch_size, go_token):
    		'''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    		ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    		dec_input = tf.concat( [tf.fill([batch_size, 1], go_token), ending], 1)
    		return dec_input
	
	@staticmethod
	def _dropout_cell(rnn_size, keep_prob):
	    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
	    return tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
	@staticmethod
	def _multi_dropout_cell(rnn_size, keep_prob, num_layers):    
	    return tf.contrib.rnn.MultiRNNCell( [Seq2Seq._dropout_cell(rnn_size, keep_prob) for _ in range(num_layers)] )
	
	
	def _encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob, sequence_lengths):
	    """
	    Create the encoding layer
	    
	    Returns a tuple `(outputs, output_states)` where
	      outputs is a 2-tuple of vectors of dimensions [sequence_length, rnn_size] for the forward and backward passes
	      output_states is a 2-tupe of the final hidden states of the forward and backward passes
	    
	    """
	    forward_cell = Seq2Seq._multi_dropout_cell(rnn_size, keep_prob, num_layers)
	    backward_cell = Seq2Seq._multi_dropout_cell(rnn_size, keep_prob, num_layers)
	    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_cell, cell_bw = backward_cell, sequence_length = sequence_lengths,
	                                                   inputs = rnn_inputs, dtype=tf.float32)
	    return outputs, states
	
	
	
	def _decoding_layer(self, enc_state, enc_outputs):
	   
	    with tf.variable_scope("decoding") as scope:
	        dec_cell = Seq2Seq._multi_dropout_cell(rnn_size, keep_prob, num_layers)
	        init_dec_state_size = batch_size
	        attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=attn_size, memory=enc_outputs, memory_sequence_length=source_lengths)
	        attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, attention_layer_size=dec_cell.output_size)
	        init_dec_state = attn_cell.zero_state(init_dec_state_size, tf.float32).clone(cell_state=enc_state)
	        
	        decoder_gen = lambda helper: tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, init_dec_state,
	                                        output_layer = output_layer)
	        
	        #TRAINING
	        train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_lengths)
	        train_decoder = decoder_gen(train_helper)
	        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, scope=scope)
	        train_logits = train_outputs.rnn_output
	
	        #INFERENCE
	        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens = tf.tile([go_token], [batch_size]), end_token = eos_token)
	        infer_decoder = decoder_gen(infer_helper)
	        infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, scope=scope, maximum_iterations=tf.round(tf.reduce_max(source_lengths) * 2))
	        infer_ids = infer_outputs.sample_id
	                
	    return train_logits, infer_ids

def vad_appended():
	placeholders = create_placeholders()
	target_lengths = placeholders.target_lengths



	
