"""
Train or query a Seq2Seq dialog generation model.
"""

#Utilities
import sys
import argparse
import pandas as pd

#Local modules
import experiment

def create_parser():
	parser = argparse.ArgumentParser(description="Train an affective neural dialog generation model."
					             " You must select one of --vad, --counter, or --retro.",
							conflict_handler="resolve")

	parser.add_argument("--vad", action="store_true", help="Model with VAD values appended to Word2Vec embeddings")
	parser.add_argument("--counter", action="store_true", help="Model with counterfitted embeddings")
	parser.add_argument("--retro", action="store_true", help="Model with retrofitted embeddings")

	parser.add_argument("--model", "-m", metavar="<path>", help="TensorFlow checkpoint path for continuing training or for inference. Note models have separate files for training and inference.")
	parser.add_argument("--infer", metavar="<path>", help="Generate responses to a text file of prompts rather than train. All prompts and their responses go to stdout. Requires --model.")
	parser.add_argument("--infer-out", metavar="<.xlsx path>", help="Write responses to Excel file rather than stdout.")

	parser.add_argument("--embeddings-only", action="store_true", help="Just generate the embeddings and exit. You may specify multiple models")
	parser.add_argument("--regen", "--regen-embeddings", action="store_true", help="Regenerate the model's embeddings prior to training")


	return parser


def beam_frame(beams):
	"""
	:param list(list(str)) beams: Set of beams for each prompt

	:returns Dataframe of the beams with columns \"beams_0\", \"beams_1\", . . . \"beams_{beam_width - 1}\"
	:rtype pd.DataFrame
	"""
	out_frame = pd.DataFrame()
	beam_width = len(beams[0])
	for i in range(beam_width):
		beam_col_i = [beam_set[i] for beam_set in beams]
		out_frame["beams_{}".format(i)] = beam_col_i
	return out_frame

def main(args):
	"""
	:param argparse.ArgumentParser args: Command-line arguments collected using argparse
	"""
	if args.embeddings_only:
		experiment.gen_embeddings(args.vad, args.counter, args.retro)
		sys.exit(0)

	if args.infer:
		exp_state = experiment.ExpState.QUERY
		regenerate = False
		if args.model is None:
			raise ValueError("Must specify --model/-m for inference")
	elif args.model:
		exp_state = experiment.ExpState.CONT_TRAIN
		regenerate = False
	else:
		exp_state = experiment.ExpState.NEW
		regenerate = args.regen


	if args.vad:
		exp = experiment.VADExp(regenerate, exp_state=exp_state, restore_path=args.model)
	elif args.counter:
		exp = experiment.Aff2VecExp(regenerate, counterfit=True, exp_state=exp_state, restore_path=args.model)
	elif args.retro:
		exp = experiment.Aff2VecExp(regenerate, counterfit=False, exp_state=exp_state, restore_path=args.model)
	else:
		parser.print_help()
		sys.exit(0)

	if args.infer is not None:
		with open(args.infer, "r", encoding="utf-8") as prompts_file:
			prompts_text = [prompt.strip() for prompt in  prompts_file.readlines()]
		beams = exp.infer(prompts_text)

		if args.infer_out:
			beam_frame(beams).to_excel(args.infer_out, index=False)
		else:
			for (i, prompt) in enumerate(prompts_text):
				sys.stdout.write("{}\n".format(prompt))
				for beam in beams[i]:
					sys.stdout.write("\t{}\n".format(beam))
	else:
		exp.train()	
	
if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()
	main(args)

