"""
Train a Seq2Seq dialog generation model.

Usage: python train.py config.yml

Mandatory YAML parameters (see :py:mod:`config`):

- vocab
- corpora
- valid_corpora
- unk
- embeddings

Recommended YAML parameters:

- arch (will soon be mandatory)
- train_save
- infer_save

"""

#Utilities
import sys

#Local modules
import config



def main(config_obj):
	"""
	Train the model

	:param config.Config config_obj: Settings for the experiment
	"""

	exp_constructor = config_obj.arch
	exp = exp_constructor(config_obj)
	exp.train()	


	
if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.stderr.write("{}\n".format(__doc__))
		sys.stderr.write("You passed in {} arguments instead of 1.\n".format(len(sys.argv) - 1))
		sys.exit(0)
	config_obj = config.Config(sys.argv[1])
	main(config_obj)

