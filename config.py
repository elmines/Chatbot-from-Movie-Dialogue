"""
Module for customizing model and experiment hyperparameters
"""
import os
import warnings
import time
import yaml

#Local modules
import experiment

class Setting(object):
	"""
	A setting/parameter that can be set in a YAML configuration file.

	:type name: str
	"""

	def __init__(self, name, type_fn, default, dependencies=None):
		"""
		:param str         name: Name of the setting
		:param callable type_fn: Function that transforms the YAML value to the desired type/object
		:param callable default: Supplier function that takes no parameters and returns a default value
		:param list(Setting) dependencies: A list of settings that must first be set before this Setting may be set
		"""
		self._name = name
		self._type_fn = type_fn
		self._default = default

		if isinstance(dependencies, Setting):
			dependencies = [dependencies]	
		self._dependencies = dependencies

	@property
	def name(self):
		"""
		str: The name of the Setting
		"""
		return self._name
	@property
	def type_fn(self):
		"""
		callable: A function with signature type_fn(yaml_value) -> converted_value
		"""
		return self._type_fn
	@property
	def default(self):
		"""
		callable: A supplier function that takes no parameters and returns a default value (still passed into type_fn later)
		"""
		return self._default
	@property
	def dependencies(self):
		"""
		list(Setting): Settings that must first be set before this Setting may be set
		"""
		return self._dependencies


def _maybe_abspath(path):
	"""
	:param path: A path-like object
	:returns An absolute path if path isn't None, otherwise None
	"""
	return os.path.abspath(path) if path is not None else path

def _maybe_str(candidate):
	"""
	:param str candidate: Either a string or None
	:returns A string if candidate isn't None, otherwise None
	"""
	return str(candidate) if candidate is not None else candidate


class Config(object):
	"""
	A namespace for the various settings of the experiment

	:ivar int                 num_layers: The number of layers (for both the encoder and decoder); default 1
	:ivar int                   rnn_size: The size of an RNN cell's hidden state; default 1024
	:ivar int                  attn_size: The size of the attention mechanism; default 256
	:ivar float            learning_rate: The step size for gradient-based optimizers; default 0.0001
	:ivar int                 beam_width: The number of beams to generate when doing decoding; default 1 (equivalent to greedy decoding)
	:ivar float      gradient_clip_value: Magnitude used to clip gradients during learning, such that gradients are clipped to [-gradient_clip_value, gradient_clip_value]; default 5.0
	:ivar int                 max_epochs: Maximum number of epochs for training; default 10
	:ivar int           train_batch_size: Minibatch size during training; default 64
	:ivar int           infer_batch_size: Minibatch size during validation, testing, etc.; default 32
	:ivar path-like           train_save: Path prefix at which to save training graphs during training; default `None`
	:ivar path-like           infer_save: Path prefix at which to save inference graphs during training; default `None`
	:ivar path-like           model_load: Path to a tf.train.Checkpoint file (used for continuing training or loading a model for inference); default `None`
	:ivar str                       arch: Type of architecture, valid values are \"vad\" (for :py:class:`experiment.VADExp`) and \"distributed\" (for :py:class:`experiment.Aff2VecExp`); default \"distributed\"
	:ivar path-like           embeddings: Path to a Numpy file (i.e. .npy or .npz) containing the embeddings for the model; default \"word_Vecs_VAD.npy\" or \"word_Vecs_retrofit_affect.npy\", depending on **arch**
	:ivar path-like                vocab: Path to a text file where each line is a word and its integer index, separated by a space
	:ivar list(path-like)        corpora: Two paths to the prompts and responses corpora for training
	:ivar list(path-like)  valid_corpora: Two paths to the prompts and responses corpora for validation
	:ivar str                        unk: The unknown token, which must be listed in **vocab**
	:ivar path-like             data_dir: Path to the directory where train_prompts.txt, ..., vocab.txt reside; default \"corpora/\"
	:ivar path-like           infer_text: Path to a text file with prompts separated by newlines; default `None`
	:ivar path-like          infer_sheet: .xlsx file where all the prompts are in a single column; default `None`
	:ivar str                  sheet_col: Column of **infer_sheet** from which to pull prompts; default `None` (caller can decide how to handle this case)
	:ivar path-like            infer_out: .xlsx file to which the model can write its responses during inference; default `None`

	"""
	def __init__(self, config_file=None):	
		"""
		:param path-like config_file: YAML configuration file

		If config_file isn't specified, `Setting.default` will be used for every Setting
		Note that most scripts require certain Settings to be explicitly specified.
		"""

		self._config_file = config_file
		self._define_settings()
		self.__setattr__ = self._setattr_base #Override __setattr__ after assigning _setting_dict

		if config_file:
			with open(config_file, "r") as r:
				yaml_dict = yaml.load(r)
			if yaml_dict is None: yaml_dict = dict() #Empty YAML file
		else:
			yaml_dict = {}

		self._initialize_settings(yaml_dict)


	def _setattr_base(self, name, value):
		#Function used to override __setattr__
		"""
		:param str name: One of the object's members
		:param value: The member's value (to be converted with a type function)
		"""
		if name not in self._setting_dict:
			raise KeyError("{} is not a valid setting.".format(name))
	
		setting = self._setting_dict[name]
		super(Config, self).__setattr__(name, setting.type_fn(value))

	def _define_settings(self):
		"""
		Subroutine that initializes a Config object's _setting_dict, which holds all its public memmbers
		"""

		_settings = []

		_timestamp = time.strftime("%b%d_%H:%M:%S")
		arch_dict = {"vad": experiment.VADExp, "distributed": experiment.DistrExp}
		def default_embedding():
			embeddings_dict = {experiment.VADExp : "word_Vecs_VAD.npy", experiment.DistrExp : "word_Vecs_retrofit_affect.npy"}
			embeddings_path = embeddings_dict[self.arch]
			warnings.warn("You did not specify the `embeddings` parameter. We are providing a default of \"{}\" for"
					" the model {}, but this behavior will be deprecated.".format(embeddings_path, self.arch))
			return embeddings_path

		#Hyperparameters
		_settings.append(Setting("num_layers",          int,             lambda: 1))
		_settings.append(Setting("rnn_size",            int,             lambda: 1024))
		_settings.append(Setting("attn_size",           int,             lambda: 256))
		_settings.append(Setting("learning_rate",       float,           lambda: 0.0001))
		_settings.append(Setting("beam_width",          int,             lambda: 1))
		_settings.append(Setting("gradient_clip_value", float,           lambda: 5.0))


		#Training
		_settings.append(Setting(      "max_epochs", int, lambda: 10))
		_settings.append(Setting("train_batch_size", int, lambda: 64))
		_settings.append(Setting("infer_batch_size", int, lambda: 32))


		#Model loading and saving
		_settings.append(Setting("train_save",          os.path.abspath, lambda: os.path.join(_timestamp, "train_model.ckpt")))
		_settings.append(Setting("infer_save",          os.path.abspath, lambda: os.path.join(_timestamp, "infer_model.ckpt")))
		_settings.append(Setting("model_load",          _maybe_abspath,  lambda: None))

		#Architecture
		arch_setting = Setting("arch",       arch_dict.__getitem__, lambda: "distributed")
		_settings.append(arch_setting)
		_settings.append(Setting("embeddings", os.path.abspath, default_embedding, dependencies=arch_setting))

		#Data files
		tuple_paths = lambda paths: tuple(os.path.abspath(path) for path in paths)
		_settings.append(Setting("vocab", os.path.abspath, lambda: None))
		_settings.append(Setting("corpora", tuple_paths, lambda: []))
		_settings.append(Setting("valid_corpora", tuple_paths, lambda: []))
		_settings.append(Setting("unk", _maybe_str, lambda: None))


		#Inference
		_settings.append(Setting("infer_text", _maybe_abspath, lambda: None))
		_settings.append(Setting("infer_sheet",   _maybe_abspath, lambda: None))
		_settings.append(Setting("sheet_col",         _maybe_str, lambda: None))
		_settings.append(Setting("infer_out",    os.path.abspath, lambda: os.path.join(_timestamp, "out.xlsx")))

		self._setting_dict = {setting.name:setting for setting in _settings}
	

	#TODO: Check for cyclical dependencies among Settings
	def _initialize_settings(self, yaml_dict):
		"""
		:param dict(str, object) yaml_dict: Dictionary read from a YAML file
		
		Initializes all the settings (i.e. public members) of self
		"""
		#Sort of a memoization table that records which settings have already been computed
		initialized = {name:False for name in self._setting_dict}
		for setting in self._setting_dict.values():
			self._initialize_setting(setting, yaml_dict, initialized)

		invalid_keys = set(yaml_dict) - set(self._setting_dict)
		if len(invalid_keys) > 0:
			raise KeyError("The following settings from {} are invalid: {}".format(self._config_file, invalid_keys))

	def _initialize_setting(self, setting, yaml_dict, initialized):
		"""
		:param Setting setting: Setting to initialize
		:param dict(object, object) yaml_dict: Dictionary read from a YAML file
		:param dict(str, bool) initialized: Dictionary indicating by name which Settings have already been initialized

		The function recurs if setting has any dependencies.
		**initialized** is used to determine when the base case has been reached.
		"""
		if initialized[setting.name]:
			return	

		if setting.dependencies:
			for dependency in setting.dependencies:
				self._initialize_setting(dependency, yaml_dict, initialized)

		if setting.name in yaml_dict:
			value = yaml_dict[setting.name]
		else:
			value = setting.default() #Calling this only if necessary allows us to put warning messages in the default functions

		self.__setattr__(setting.name, yaml_dict.get(setting.name, value))
		initialized[setting.name] = True

