"""
Module for customizing model and experiment hyperparameters
"""
import yaml
from collections import namedtuple

class Setting(namedtuple("Setting", ["name", "type_fn", "default"])):
	pass

class Config(object):

	_settings = []
	_settings.append(Setting("num_layers",      int, 1))
	_settings.append(Setting("learning_rate", float, 0.001))

	_setting_dict = {setting.name:setting for setting in _settings}


	def __init__(self, config_file):	
		with open(config_file, "r") as r:
			yaml_dict = yaml.load(r)

		consumed_keys = set()
		for setting in Config._settings:
			name = setting.name
			__setattr__(self, name, yaml_dict.get(name, setting.default))
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
	print("Called __setattr__", flush=True)
	if name not in self._setting_dict:
		raise KeyError("{} is not a valid setting.".format(name))

	setting = Config._setting_dict[name]
	super(Config, self).__setattr__(name, setting.type_fn(value))

if __name__ == "__main__":
	config = Config("config.yml")
	#print(config.num_layers)
	print(config.learning_rate)
	print(type(config.learning_rate))

	config.ip_address = 5
	print(config.ip_address)
