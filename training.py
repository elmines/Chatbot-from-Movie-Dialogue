import tensorflow as tf
import numpy as np


###########SAMPLLING OUTPUT##############
#FIXME: padding parameters
def show_response(prompt_int, prediction, answer_int = None):
    pad_q = METATOKEN_INDEX
    print("Prompt")
    print("  Word Ids: {}".format([i for i in prompt_int if i != pad_q]))
    print("      Text: {}".format(int_to_text(prompt_int, prompts_int_to_vocab)))
    
    pad_a = METATOKEN_INDEX
    if answer_int is not None:
        print("Target Answer")
        print("  Word Ids: {}".format([i for i in answer_int if i != pad_a]))
        print("      Text: {}".format(int_to_text(answer_int, answers_int_to_vocab)))

    print("\nPrediction")
    print('  Word Ids: {}'.format([i for i in prediction if i != pad_a]))
    print('      Text: {}'.format(int_to_text(prediction, answers_int_to_vocab)))
        
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


############FEEDDING DATA################
#FIXME: padding parameters
def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    pad_int = METATOKEN_INDEX
    max_sentence_length = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence_length - len(sentence)) for sentence in sentence_batch]

#FIXME: padding parameters
def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        
        source_lengths = np.array( [len(sentence) for sentence in questions_batch] )
        target_lengths = np.array( [len(sentence) for sentence in answers_batch])
        
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        yield source_lengths, target_lengths, pad_questions_batch, pad_answers_batch

def parallel_shuffle(source_sequences, target_sequences):
    if len(source_sequences) != len(target_sequences):
        raise ValueError("Cannot shuffle parallel sets with different numbers of sequences")
    indices = np.random.permutation(len(source_sequences))
    shuffled_source = [source_sequences[indices[i]] for i in indices]
    shuffled_target = [target_sequences[indices[i]] for i in indices]
    
    return (shuffled_source, shuffled_target)

############LOGGING###################
def log_entries(csv_path, *fields, header = False):
    if len(fields[0]) < 1: return
    mode = "w" if header else "a"
    with open(csv_path, mode, encoding="utf-8") as log:
        lines = []
        num_lines = len(fields[0])
        lines = "\n".join(",".join([str(field[i]) for field in fields]) 
                          for i in range(num_lines)
        )
        log.write(lines + "\n")
def clear_fields(log_fields):
    for field in log_fields:
        field.clear()


def training_loop(epoch_no, max_epochs, placeholders, train_op, train_loss, valid_loss, saver, best_valid_loss=float("inf"), best_model_path=None, latest_model_path=None, train_log=None, valid_log=None, new_logs=False, early_stopping=True):
	"""
	placeholders - a models.Placeholders named tuple
	train_op - Train operation to be called with sess.run
	train_loss - Scalar loss node to be computed with sess.run
	valid_loss - Scalar loss node to be computed with sess.run
	Returns
		- (float) The number of epochs spent training
		- (float) The best validation loss computed
	"""

	train_epoch_nos = []
	train_batch_tokens = [] #Number of tokens in a batch
	train_batch_losses = [] #Per-token loss for a batch
	train_log_fields = [train_epoch_nos, train_batch_tokens, train_batch_losses]
	
	valid_epoch_nos = []
	valid_check_nos = []
	valid_batch_tokens = []
	valid_batch_losses = []
	valid_log_fields = [valid_epoch_nos, valid_check_nos, valid_batch_tokens, valid_batch_losses]
	
	train_start_time = None #For logging time for sets of batches
	if new_logs:
		log_entries(train_log, ["epoch"], ["num_tokens"], ["loss_per_token"], header=True)
	log_entries(valid_log, ["epoch"], ["check"], ["num_tokens"], ["loss_per_token"], header=True)
	print("Initialized empty training log {}, validation log {}".format(train_log, valid_log))
	
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    tf.train.Saver().save(sess, checkpoint_latest)
	    print("Initialized model parameters, wrote initial model to {}".format(checkpoint_latest))
	    if not use_affect_func: print("Beginning training with cross-entropy loss.")
	    else:                   print("Beginning training with {}".format(affect_function))
	    for epoch_i in range(1, epochs+1):
	        if not use_affect_func and epoch_i > epochs_before_affective_loss:
	            print("Switching from cross-entropy loss to {}".format(affect_function))
	            use_affect_func = True        
	        
	        print("Shuffling training data . . .")
	        (train_prompts_int, train_answers_int) = parallel_shuffle(train_prompts_int, train_answers_int)
	        
	        valid_check_no = 1
	        
	        
	        for batch_i, (p_lengths, a_lengths, prompts_batch, answers_batch) in enumerate(
	                batch_data(train_prompts_int, train_answers_int, train_batch_size)):
	            if train_start_time is None: train_start_time = time.time()
	            
	            #VALIDATION CHECK
	            if batch_i % validation_check == 0 and epoch_i > min_epochs_before_validation:
	                print("Shuffling validation data . . .")
	                (valid_prompts_int, valid_answers_int) = parallel_shuffle(valid_prompts_int, valid_answers_int)
	                
	                clear_fields(valid_log_fields)
	
	                
	                valid_start_time = time.time()
	                for batch_ii, (p_lengths, a_lengths, prompts_batch, answers_batch) in \
	                        enumerate(batch_data(valid_prompts_int, valid_answers_int, valid_batch_size)):
	
	                    [valid_loss] = sess.run([perplexity],
	                        {input_data: prompts_batch, targets: answers_batch,
	                        source_lengths: p_lengths, target_lengths: a_lengths, keep_prob: 1})
	                    valid_epoch_nos.append(epoch_i)
	                    valid_check_nos.append(valid_check_no)
	                    valid_batch_tokens.append(sum(a_lengths))
	                    valid_batch_losses.append(valid_loss)
	
	                
	                valid_check_no += 1
	                duration = time.time() - valid_start_time
	                avg_valid_loss = sum(loss*tokens 
	                        for (loss, tokens) in zip(valid_batch_losses, valid_batch_tokens)) / sum(valid_batch_tokens)
	                
	                log_entries(valid_log, *(valid_log_fields))
	                clear_fields(valid_log_fields)
	                print("Processed validation set in {:>4.2f} seconds".format(duration))
	                print("Average perplexity per token = {}".format(avg_valid_loss))
	                if avg_valid_loss >= best_valid_loss:
	                    print("No improvement for validation loss.")
	                else:
	                    best_valid_loss = avg_valid_loss
	                    print("New record for validation loss!")
	                    print("Saving best model to {}".format(checkpoint_best))
	                    tf.train.Saver().save(sess, checkpoint_best)
	                check_response(sess, prompts_batch[-1], answers_batch[-1])
	                
	                train_start_time = time.time()
	            
	            #TRAINING
	            _, loss = sess.run([train_op, train_cost],
	                {input_data: prompts_batch, targets: answers_batch,
	                 source_lengths: p_lengths, target_lengths: a_lengths,
	                 keep_prob: keep_probability,
	                 train_affect: use_affect_func})
	            train_epoch_nos.append(epoch_i)
	            train_batch_losses.append(loss)
	            train_batch_tokens.append(sum(a_lengths))
	            
	            if batch_i % display_step == 0:
	                duration = time.time() - train_start_time
	                avg_train_loss = sum(loss*tokens 
	                        for (loss, tokens) in zip(train_batch_losses, train_batch_tokens)) / sum(train_batch_tokens)
	                    
	                log_entries(train_log, *(train_log_fields))
	                clear_fields(train_log_fields)
	                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss-per-Token: {:>9.6f}, Seconds: {:>4.2f}'
	                      .format(epoch_i, epochs, batch_i, len(train_prompts_int) // train_batch_size, 
	                              avg_train_loss, duration),
	                         flush=True)
	                train_start_time = time.time()
	
	        print("{} epochs completed, saving model to {}".format(epoch_i, checkpoint_latest))
	        tf.train.Saver().save(sess, checkpoint_latest)
	        log_entries(train_log, *(train_log_fields))
	        clear_fields(train_log_fields)
