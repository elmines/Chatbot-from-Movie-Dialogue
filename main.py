"""
Train a Seq2Seq dialog generation model.

Usage: `python config.yml`
"""

#Utilities
import sys

#Local modules
import config


#def beam_frame(beams):
	#"""
	#:param list(list(str)) beams: Set of beams for each prompt
#
	#:returns Dataframe of the beams with columns \"beams_0\", \"beams_1\", . . . \"beams_{beam_width - 1}\"
	#:rtype pd.DataFrame
	#"""
	#out_frame = pd.DataFrame()
	#beam_width = len(beams[0])
	#for i in range(beam_width):
		#beam_col_i = [beam_set[i] for beam_set in beams]
		#out_frame["beams_{}".format(i)] = beam_col_i
	#return out_frame

def main(config_obj):
	"""
	:param argparse.ArgumentParser args: Command-line arguments collected using argparse
	"""

	exp_constructor = config_obj.arch
	exp = exp_constructor(config_obj)
	exp.train()	

	#if args.infer is not None:
		#with open(args.infer, "r", encoding="utf-8") as prompts_file:
			#prompts_text = [prompt.strip() for prompt in  prompts_file.readlines()]
		#beams = exp.infer(prompts_text)
#
		#if args.infer_out:
			#beam_frame(beams).to_excel(args.infer_out, index=False)
		#else:
			#for (i, prompt) in enumerate(prompts_text):
				#sys.stdout.write("{}\n".format(prompt))
				#for beam in beams[i]:
					#sys.stdout.write("\t{}\n".format(beam))

	
if __name__ == "__main__":
	if len(sys.argv) < 2:
		sys.stderr.write("{}\n".format(__doc__))
		sys.exit(0)
	config_obj = config.Config(sys.argv[1])
	main(config_obj)

