"""
Module for customizing model and experiment hyperparameters
"""
import yaml
from collections import namedtuple
import time
import os

class Setting(namedtuple("Setting", ["name", "type_fn", "default"])):
	"""
	str name: Name of the setting
	callable type_fn: Function that transforms the YAML value to the desired type/object
	callable default: Supplier function that takes no parameters and returns a default value
	"""
	pass

def _maybe_abspath(path):
	"""
	:param path: A path-like object
	:returns An absolute path if path isn't None, otherwise None
	"""
	return os.path.abspath(path) if path is not None else path


class Config(object):

	def __init__(self, config_file=None):	
		self._setting_dict = Config._define_settings()
		self.__setattr__ = self._setattr_base #Override __setattr__ after assigning _setting_dict

		if config_file:
			with open(config_file, "r") as r:
				yaml_dict = yaml.load(r)
			if yaml_dict is None: yaml_dict = dict() #Empty YAML file
		else:
			yaml_dict = {}

		consumed_keys = set()
		for setting in self._setting_dict.values():
			name = setting.name
			print(name)
			self.__setattr__(name, yaml_dict.get(name, setting.default()))
			if name in yaml_dict:
				consumed_keys.add(name)

		invalid_keys = set(yaml_dict) - consumed_keys
		if len(invalid_keys) > 0:
			raise KeyError("The following settings from {} are invalid: {}".format(config_file, invalid_keys))


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

	@staticmethod	
	def _define_settings():
		_timestamp = time.strftime("%b%d_%H:%M:%S")

		_settings = []
		_settings.append(Setting("num_layers",          int,             lambda: 1))
		_settings.append(Setting("rnn_size",            int,             lambda: 1024))
		_settings.append(Setting("attn_size",           int,             lambda: 256))
		_settings.append(Setting("learning_rate",       float,           lambda: 0.0001))
		_settings.append(Setting("beam_width",          int,             lambda: 1))
		_settings.append(Setting("gradient_clip_value", float,           lambda: 5.0))
		_settings.append(Setting("train_save",          os.path.abspath, lambda: os.path.join(_timestamp, "train_model.ckpt")))
		_settings.append(Setting("infer_save",          os.path.abspath, lambda: os.path.join(_timestamp, "infer_model.ckpt")))

		_settings.append(Setting("model_load",          _maybe_abspath,  lambda: None))

		_settings.append(Setting("infer_prompts", _maybe_abspath, lambda: None))
	
		_setting_dict = {setting.name:setting for setting in _settings}
		return _setting_dict
	




if __name__ == "__main__":
	#config = Config("config.yml")
	config = Config()
	print(config.num_layers)
	print(config.learning_rate)

	#config.learning_rate = "0.004"
	#print(config.learning_rate)
	#print(type(config.learning_rate))

