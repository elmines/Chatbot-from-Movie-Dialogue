#!/bin/bash
import sys
if ".." not in sys.path: sys.path.append("..")

from corpus import Corpus

my_corp = Corpus("./movie_lines.txt", "movie_conversations.txt",
		max_vocab = 12000)

my_corp.write_prompts("prompts.txt")
my_corp.write_answers("answers.txt")
my_corp.write_vocab("vocab.txt")
