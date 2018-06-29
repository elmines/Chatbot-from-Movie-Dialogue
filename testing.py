import numpy as np



def trunc_padding(sequence, pad_token):
        return [token for token in sequence if token != pad_token]

def int_to_text(sequence, int2vocab):
        return [int2vocab[index] for index in sequence]

def text_to_int(sequence, vocab2int, unk_token):
        return [vocab2int.get(token, unk_token) for token in sequence if token not in codes]

###########SAMPLLING OUTPUT##############
def show_response(prompt_int, prediction_int, int_to_vocab, pad_q, pad_a, answer_int = None):
        print("Prompt")
        print("  Word Ids: {}".format([i for i in prompt_int if i != pad_q]))
        print("      Text: {}".format(int_to_text(prompt_int, prompts_int_to_vocab)))
        if answer_int is not None:
                print("Target Answer")
                print("  Word Ids: {}".format([i for i in answer_int if i != pad_a]))
                print("      Text: {}".format(int_to_text(answer_int, answers_int_to_vocab)))
        print("\nPrediction")
        print('  Word Ids: {}'.format([i for i in prediction_int if i != pad_a]))
        print('      Text: {}'.format(int_to_text(prediction_int, answers_int_to_vocab)))
        
#FIXME: padding parameters
def check_response(session, prompt_int, answer_int=None):
    """
    session - the TensorFlow session
    question_int - a list of integers
    answer - the actual, correct response (if available)
    """
    
    two_d_prompt_int = [prompt_int]
    p_lengths = [len(prompt_int)]
    
    [infer_ids_output] = session.run([infer_ids], feed_dict = {input_data: np.array(two_d_prompt_int, dtype=np.float32),
                                                      source_lengths: p_lengths,
                                                      keep_prob: 1})
    
    show_response(prompt_int, infer_ids_output[0], answer_int)

