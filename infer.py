"""
Script for inferring responses to human prompts using a Seq2Seq dialog generation model

Usage: python infer.py config.yml

Required YAML parameters (see :py:mod:`config`):

- vocab
- corpora
- valid_corpora
- unk
- embeddings
- model_load
- arch

Recommended YAML parameters:

- infer_sheet
- infer_col
- infer_out

"""

#Utilities
import sys
import os

#Data science
import pandas as pd

#Local modules
import config


def _extract_prompts(config_obj):
	"""
	:param config.Config config_obj: The configuration for the experiment

	:returns The prompts, either from a spreadsheet, a text file, or stdin
	:rtype list(str)
	"""
	if not(config_obj.infer_text or config_obj.infer_sheet):
		sys.stderr.write("No prompts provided with YAML parameters `infer_text` or `infer_sheet`.\n"
               		         "Reading prompts from standard input . . .\n")
		prompts_text = []
		prompt = sys.stdin.readline().strip()
		while prompt:
			prompts_text.append(prompt)
			prompt = sys.stdin.readline().strip()
	elif config_obj.infer_text:
		if config_obj.infer_sheet:
			sys.stderr.write("Provided both `infer_text` and `infer_sheet`. Ignoring `infer_sheet` . . .\n")
		with open(config_obj.infer_text, "r", encoding="utf-8") as prompts_file:
			prompts_text = [prompt.strip() for prompt in  prompts_file.readlines()]
	else:
		df = pd.read_excel(config_obj.infer_sheet)
		if not config_obj.sheet_col:
			sheet_col = df.columns[0]
			sys.stderr.write("YAML parameter `sheet_col` not specified--using first column {}\n".format(sheet_col))
		else:
			sheet_col = config_obj.sheet_col

		prompts_text = [prompt.strip() for prompt in df[sheet_col]]

	return prompts_text

def main(config_obj):
	"""
	Evaluate the model

	:param config.Config config_obj: Settings for the experiment
	"""
	if not config_obj.model_load:
		raise ValueError("Must specify `model_load` YAML parameter when performing inference.")
	exp_constructor = config_obj.arch
	exp = exp_constructor(config_obj, infer=True)

	prompts_text = _extract_prompts(config_obj)
	df_responses = exp.infer(prompts_text)

	sys.stderr.write("Writing all responses to spreadsheet {}\n".format(config_obj.infer_out))
	out_dir = os.path.dirname(config_obj.infer_out)
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	df_responses.to_excel(config_obj.infer_out, index=False)


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.stderr.write("{}\n".format(__doc__))
		sys.stderr.write("You passed in {} arguments instead of 1.\n".format(len(sys.argv) - 1))
		sys.exit(0)
	config_obj = config.Config(sys.argv[1])
	main(config_obj)

