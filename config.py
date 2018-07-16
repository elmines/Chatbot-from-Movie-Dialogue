"""
Module for customizing model and experiment hyperparameters
"""
import yaml
from collections import namedtuple

class Setting(namedtuple("Setting", ["name", "type_fn", "default"])):
	"""
	str name: Name of the setting
	callable type_fn: Function that transforms the YAML value to the desired type/object
	callable default: Supplier function that returns a default value when called
	"""
	pass

class Config(object):

	_settings = []
	_settings.append(Setting("num_layers",      int, lambda: 1))
	_settings.append(Setting("rnn_size",        int, lambda: 1024))
	_settings.append(Setting("attn_size",       int, lambda: 256))
	#_settings.append(Setting("learning_rate", float, lambda: 0.0001))

	_setting_dict = {setting.name:setting for setting in _settings}


	def __init__(self, config_file=None):	
		if config_file:
			with open(config_file, "r") as r:
				yaml_dict = yaml.load(r)
			if yaml_dict is None: yaml_dict = dict() #Empty YAML file
		else:
			yaml_dict = {}

		consumed_keys = set()
		for setting in Config._settings:
			name = setting.name
			self.__setattr__(name, yaml_dict.get(name, setting.default()))
			if name in yaml_dict:
				consumed_keys.add(name)

		invalid_keys = set(yaml_dict) - consumed_keys
		if len(invalid_keys) > 0:
			raise KeyError("The following settings from {} are invalid: {}".format(config_file, invalid_keys))


	def __setattr__(self, name, value):
		"""
		:param str name: One of the object's members
		:param str value: The member's value (to be converted with a type function)
	
		"""
		if name not in self._setting_dict:
			raise KeyError("{} is not a valid setting.".format(name))
	
		setting = Config._setting_dict[name]
		super(Config, self).__setattr__(name, setting.type_fn(value))

if __name__ == "__main__":
	#config = Config("config.yml")
	config = Config()
	print(config.num_layers)
	print(config.learning_rate)

	#config.learning_rate = "0.004"
	#print(config.learning_rate)
	#print(type(config.learning_rate))

