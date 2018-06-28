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


def _dropout_cell(rnn_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        return tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
def _multi_dropout_cell(rnn_size, keep_prob, num_layers):    
        return tf.contrib.rnn.MultiRNNCell( [Seq2Seq._dropout_cell(rnn_size, keep_prob) for _ in range(num_layers)] )
def _encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_lengths):
        """
        Create the encoding layer
	    
        Returns a tuple `(outputs, output_states)` where
                outputs is a 2-tuple of vectors of dimensions [sequence_length, rnn_size] for the forward and backward passes
                output_states is a 2-tupe of the final hidden states of the forward and backward passes
        """
        forward_cell = _multi_dropout_cell(rnn_size, keep_prob, num_layers)
        backward_cell = _multi_dropout_cell(rnn_size, keep_prob, num_layers)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = forward_cell, cell_bw = backward_cell, sequence_length = sequence_lengths,
	                                                   inputs = rnn_inputs, dtype=tf.float32)
        return outputs, states
def _process_decoding_input(target_data, go_token):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        batch_size = tf.shape(target_data)[0]
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat( [tf.fill([batch_size, 1], go_token), ending], 1)
        return dec_input
def _decoding_layer(enc_state, enc_outputs, dec_embed_input, dec_embeddings, num_layers, rnn_size, attn_size, output_layer, keep_prob, source_lengths, target_lengths, go_token, eos_token):
        batch_size = tf.shape(source_lengths)[0]

        dec_cell = _multi_dropout_cell(rnn_size, keep_prob, num_layers)
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=attn_size, memory=enc_outputs, memory_sequence_length=source_lengths)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, attention_layer_size=dec_cell.output_size)

        init_attn_dec_state = attn_cell.zero_state(batch_size, tf.float32).clone(cell_state=enc_state)
        
        decoder_gen = lambda helper: tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, init_attn_dec_state,
        output_layer = output_layer)
        
        #TRAINING
        train_helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_lengths)
        train_decoder = decoder_gen(train_helper)
        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True) #FIXME: Add back reusable variable scope?
        train_logits = train_outputs.rnn_output
        
        #INFERENCE
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens = tf.tile([go_token], [batch_size]), end_token = eos_token)
        infer_decoder = decoder_gen(infer_helper)
        infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, impute_finished=True, maximum_iterations=tf.round(tf.reduce_max(source_lengths) * 2))
        infer_ids = infer_outputs.sample_id
        
        return train_logits, infer_ids


class Seq2Seq(object):

	def __init__(self, placeholders, enc_embedddings, dec_embeddings, go_token, eos_token,
			num_layers=1, rnn_size=1024, attn_size=256, output_layer=None,
			learning_rate=0.0001):
		"""
		placeholders - A Placeholders namedtuple
		learning_rate - A constant (you can't decay it over time)
		"""

		self.input_data = placeholders.input_data
		self.targets = placeholders.targets
                self.keep_prob = placeholders.keep_prob

                #Both of these could be necessary for subclasses with custom loss functions
                self._source_lengths = placeholders.source_lengths
                self._target_lengths = placeholders.target_lengths
		self.enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, self.input_data)
		self.dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, _process_decoding_input(targets, go_token)


		enc_outputs, enc_states = _encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, source_lengths)    
    		concatenated_enc_output = tf.concat(enc_outputs, -1)
    		init_dec_state = enc_states[0] 
    
    
    		self._train_logits, self._infer_ids = _decoding_layer(init_dec_state, concatenated_enc_output, dec_embed_input, dec_embeddings,
                                num_layers, rnn_size, attn_size, output_layer, self.keep_prob, self._source_lengths, self._target_lengths, go_token, eos_token)

		self.eval_mask = tf.sequence_mask(target_lengths, dtype=tf.float32)
		self.xent = tf.contrib.seq2seq.sequence_loss(self._train_logits, self._targets, self.eval_mask)

		self._optimizer = tf.train.AdamOptimizer(learning_rate)

		

	def train(self):
		raise NotImplementedError("Abstract method")


	@property
	def train_logits(self):
		return self._train_logits

	@property
	def infer_ids(self):
		return self._infer_ids

        @property
        def optimizer(self):
                return self._optimizer

class VADAppended(Seq2Seq):

	def __init__(self, placeholders, full_embeddings, go_token, eos_token,
                num_layers=1, rnn_size=1024, attn_size=256, output_layer=None, affect_strength=0.5):
		"""
		affect_strength - hyperparameter in the range [0.0, 1.0)
		"""
		
		Seq2Seq.__init__(self, placeholders, word_embeddings, word_embeddings,go_token, eos_token,
					num_layers, rnn_size, attn_size, output_layer)

		emot_embeddings = full_embeddings[-3: ]

            	neutral_vector = tf.constant([5.0, 1.0, 5.0], dtype=tf.float32)
            	affective_loss = loss_functions.max_affective_content(affect_strength, self._train_logits, self.targets, emot_embeddings, neutral_vector, self.eval_mask)


		self._train_affect = tf.Variable(False, trainable=False, dtype=tf.bool)


		train_cost = tf.cond(self._train_affect, true_fn= lambda: affective_loss, false_fn= lambda: self.xent)
		gradients = self.optimizer.compute_gradients(train_cost)
		capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
		self.train_op = optimizer.apply_gradients(capped_gradients)

        @property
        def train_affect(self):
                return self._train_affect
