"""
Namespace classes for organizing data
"""
import collections

class DataPlaceholders(collections.namedtuple("DataPlaceholders", ["input_data", "targets", "source_lengths", "target_lengths"])):
	"""
	`collections.namedtuple` class for holding the 4 placeholder varibles that should be fed when training a model

	:ivar tf.Tensor     input_data: TensorFlow placeholder variable for the input data
	:ivar tf.Tensor        targets: TensorFlow placeholder variable for the target labels
	:ivar tf.Tensor source_lengths: TensorFlow placeholder variable for the lengths of source sequences
	:ivar tf.Tensor target_lengths: TensorFlow placeholder variable for the lengths of target sequences
	"""	
	pass

