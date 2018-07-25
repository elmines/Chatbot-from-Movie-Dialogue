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
	"""

	def __init__(self, name, type_fn, default, dependencies=None):
		"""
		:param str         name: Name of the setting
		:param callable type_fn: Function that transforms the YAML value to the desired type/object
		:param callable default: Supplier function that takes no parameters and returns a default value
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
		The name of the Setting
		"""
		return self._name
	@property
	def type_fn(self):
		"""
		A function with signature type_fn(yaml_value) -> converted_value
		"""
		return self._type_fn
	@property
	def default(self):
		"""
		A callable that takes no parameters and returns a default value (still passed into type_fn later)
		"""
		return self._default
	@property
	def dependencies(self):
		"""
		A list of Setting objects that must first be set before this Setting may be set
		"""
		return self._dependencies


def _maybe_abspath(path):
	"""
	:param path: A path-like object
	:returns An absolute path if path isn't None, otherwise None
	"""
	return os.path.abspath(path) if path is not None else path

def _maybe_str(candidate):
	return str(candidate) if candidate is not None else candidate


class Config(object):

	def __init__(self, config_file=None):	
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
		:param str value: The member's value (to be converted with a type function)
	
		"""
		if name not in self._setting_dict:
			raise KeyError("{} is not a valid setting.".format(name))
	
		setting = self._setting_dict[name]
		super(Config, self).__setattr__(name, setting.type_fn(value))

	def _define_settings(self):

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
		_settings.append(Setting("data_dir", os.path.abspath,     lambda: os.path.abspath("corpora/")))

		#Inference
		_settings.append(Setting("infer_text", _maybe_abspath, lambda: None))
		_settings.append(Setting("infer_sheet",   _maybe_abspath, lambda: None))
		_settings.append(Setting("sheet_col",         _maybe_str, lambda: None))
		_settings.append(Setting("infer_out",    os.path.abspath, lambda: os.path.join(_timestamp, "out.xlsx")))

		self._setting_dict = {setting.name:setting for setting in _settings}
	

	#TODO: Check for cyclical dependencies among Settings
	def _initialize_settings(self, yaml_dict):
		#Sort of a memoization table that records which settings have already been computed
		initialized = {name:False for name in self._setting_dict}
		for setting in self._setting_dict.values():
			self._initialize_setting(setting, yaml_dict, initialized)

		invalid_keys = set(yaml_dict) - set(self._setting_dict)
		if len(invalid_keys) > 0:
			raise KeyError("The following settings from {} are invalid: {}".format(self._config_file, invalid_keys))

	def _initialize_setting(self, setting, yaml_dict, initialized):
		"""
		:param Setting setting
		:param dict(object, object) yaml_dict
		:param dict(str, bool) initialized
		"""
		if initialized[setting.name]:
			return	

		if setting.dependencies:
			for dependency in setting.dependencies:
				self._initialize_setting(dependency, yaml_dict, initialized)

		self.__setattr__(setting.name, yaml_dict.get(setting.name, setting.default()))
		initialized[setting.name] = True


if __name__ == "__main__":
	#config = Config("config.yml")
	config = Config()
	print(config.num_layers)
	print(config.learning_rate)

	#config.learning_rate = "0.004"
	#print(config.learning_rate)
	#print(type(config.learning_rate))

