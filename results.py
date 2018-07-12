"""
Crude wrapper script for main.py that concatentates inferences to the test set from testset.py
"""
import sys
import pandas as pd

import main
import experiment

if __name__ == "__main__":
	parser = main.create_parser()
	#Override existing --infer argument
	parser.add_argument("--infer", required=True, metavar="<.xlsx path>", help="Test set generated using preprocessing.testset.")
	args = parser.parse_args()
	assert args.infer is not None
	assert args.model is not None
	assert args.infer_out is not None

	exp_state = experiment.ExpState.QUERY
	regenerate=False
	if args.vad:
		exp = experiment.VADExp(regenerate, exp_state=exp_state, restore_path=args.model)
	elif args.counter:
		exp = experiment.Aff2VecExp(regenerate, counterfit=True, exp_state=exp_state, restore_path=args.model)
	elif args.retro:
		exp = experiment.Aff2VecExp(regenerate, counterfit=False, exp_state=exp_state, restore_path=args.model)
	else:
		parser.print_help()
		sys.exit(0)

	df_base = pd.read_excel(args.infer)
	prompts_text = [prompt.strip() for prompt in df_base["target_questions"]]
	beams = exp.infer(prompts_text)
	df_beams = main.beam_frame(beams)


	df_out = pd.concat([df_base, df_beams], axis="columns")
	df_out.to_excel(args.infer_out, index=False)

	
