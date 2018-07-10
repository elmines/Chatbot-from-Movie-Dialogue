"""
Train or query a Seq2Seq dialog generation model.
"""

#Utilities
import sys
import os
import argparse

#Only here so the seed can be set
import tensorflow as tf
import numpy as np
SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#Local modules
import experiment

def create_parser():
	parser = argparse.ArgumentParser(description="Train an affective neural dialog generation model")

	parser.add_argument("--embeddings-only", action="store_true", help="Just generate the embeddings for the specified model(s) and exit")
	parser.add_argument("--infer", nargs=2, metavar="<path>", help="TensorFlow checkpoint path and path to text file of prompts")

	parser.add_argument("--regen-embeddings", action="store_true", help="Regenerate affective embeddings prior to training")

	parser.add_argument("--vad", action="store_true", help="Train the model with VAD values appended to Word2Vec embeddings")
	parser.add_argument("--counter", action="store_true", help="Train the model with counterfitted embeddings")
	parser.add_argument("--retro", action="store_true", help="Train the model with retrofitted embeddings")

	return parser


	
if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()
	if args.embeddings_only:
		experiment.gen_embeddings(args.vad, args.counter, args.retro)
		sys.exit(0)

	regenerate = False if args.infer is not None else args.regen_embeddings
	exp_state = experiment.ExpState.QUERY if args.infer else experiment.ExpState.NEW
	if args.vad:
		exp = experiment.VADExp(regenerate, exp_state=exp_state)
	elif args.counter:
		exp = experiment.Aff2VecExp(regenerate, counterfit=True, exp_state=exp_state)
	elif args.retro:
		exp = experiment.Aff2VecExp(regenerate, counterfit=False, exp_state=exp_state)
	else:
		parser.print_help()
		sys.exit(0)

	if args.infer is not None:
		with open(args.infer[1], "r", encoding="utf-8") as prompts_file:
			prompts_text = prompts_file.readlines()
		exp.infer(args.infer[0], prompts_text)
	else:
		exp.train()
