import collections

class DataPlaceholders(collections.namedtuple("DataPlaceholders", ["input_data", "targets", "source_lengths", "target_lengths"])):
	pass

class Datasets(collections.namedtuple("Datasets", ["train_prompts_int", "train_answers_int", "valid_prompts_int", "valid_answers_int"]))
        pass


class TextData(collections.namedtuple("Text", ["int_to_vocab", "unk_int", "eos_int", "pad_int"]))
        pass
