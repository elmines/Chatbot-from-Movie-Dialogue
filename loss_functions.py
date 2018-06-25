import tensorflow as tf
import nltk
import numpy as np

def cross_entropy(train_logits,targets,eval_mask):
    xent_loss = tf.contrib.seq2seq.sequence_loss(train_logits, targets, eval_mask)
    return xent_loss

def embed_predicted_timestep(predictions):
        """
        predictions - Tensor of shape [batch_size, max_time, embedding_size]

        Returns
        averages - A Tensor of shape [batch_size, max_time, embedding_size]
                   where averages[i][j] is the average embedding for sample i
                   over the subsequence spanning the first j+1 timesteps
        """
        max_time = tf.shape(predictions)[1]
        divisors = tf.range(1, limit=max_time+1)
        sums = tf.cumsum(predictions, axis = 1)
        #tf.expand_dims necessary for proper broadcasting across the embedding axis
        averages = sums / tf.cast( tf.expand_dims(divisors, axis=-1), tf.float32) 
        return averages
    
    
def reduce_gather(params, indices):
    """
       params - An order-K Tensor
       indices - An order-{K-1} integer Tensor whose shape is equal to tf.shape(params)[:-1]
       Returns
       output - order-{K-1} Tensor where output[i, j] = params[i, j, indices[i, j]]
    """

    flat_params = tf.reshape(params, [-1, tf.shape(params)[-1]]) #Flatten everything but last axis
    flat_indices = tf.cast(tf.reshape(indices, [-1]), tf.int32)

    row_indices = tf.range(0, tf.shape(flat_params)[0])
    augmented_indices = tf.stack( [row_indices, flat_indices], axis = -1)

    elements = tf.gather_nd(flat_params, augmented_indices)

    return tf.reshape(elements, tf.shape(params)[:-1])


def _average(losses, weights, across_timesteps, across_batch):

    if not (across_timesteps or across_batch): return losses
    
    axes = []
    if across_batch: axes.append(0)
    if across_timesteps: axes.append(1)
        
    loss_sums = tf.reduce_sum(losses, axis=axes)
    loss_counts = tf.reduce_sum(weights, axis=axes)
    loss_counts += 1e-12  # to avoid division by 0 for all-0 weights
    loss_averages = loss_sums / loss_counts

    return loss_averages

def _affective_dissonance(sign, lambda_param,logits,targets,enc_embed_input,full_embeddings,weights = None, average_across_timesteps=True, average_across_batch=True):
    param = lambda_param
    param_inv = 1.0 - lambda_param

    softmaxed_logits = tf.nn.softmax(logits)
    xent = -tf.log(reduce_gather(softmaxed_logits,targets))


    input_embed= tf.reduce_mean(enc_embed_input,axis=1, keepdims=True)
    predicted_ids = tf.argmax(softmaxed_logits,axis=-1)
    prediction_vectors = tf.nn.embedding_lookup(full_embeddings, predicted_ids)
    average_prediction_vectors = embed_predicted_timestep(prediction_vectors)
    affect_dissonance = tf.norm(input_embed - average_prediction_vectors, axis=-1)
    
    predicted_prob = tf.reduce_max(softmaxed_logits,axis=-1)
    losses = (param_inv*xent) + (sign)*(param*predicted_prob*affect_dissonance)
    if weights is not None: losses *= weights
    losses = _average(losses, weights, average_across_timesteps, average_across_batch)  
    return losses

def min_affective_dissonance(lambda_param,logits,targets,enc_embed_input, embeddings,weights = None, average_across_timesteps=True, average_across_batch=True):
    return _affective_dissonance(1, lambda_param, logits, targets, enc_embed_input, embeddings, weights, average_across_timesteps, average_across_batch)

def max_affective_dissonance(lambda_param,logits,targets,enc_embed_input, embeddings,weights = None, average_across_timesteps=True, average_across_batch=True):
    return _affective_dissonance(-1, lambda_param, logits, targets, enc_embed_input, embeddings, weights, average_across_timesteps, average_across_batch)

def max_affective_content(lambda_param,logits, targets,embeddings, neutral_vector, 
                          weights = None, average_across_timesteps=True, average_across_batch=True):
    param = lambda_param
    param_inv = 1.0 - param
    softmaxed_logits = tf.nn.softmax(logits)
    xent = -tf.log(reduce_gather(softmaxed_logits,targets))

    
    predicted_ids = tf.argmax(softmaxed_logits,axis=-1)
    prediction_vectors = tf.nn.embedding_lookup(embeddings, predicted_ids)
    
    affect_dissonance = tf.norm(prediction_vectors - neutral_vector, axis=-1)
    
    predicted_prob = tf.reduce_max(softmaxed_logits,axis=-1)
    losses = (param_inv*xent) - (param*predicted_prob*affect_dissonance)
    if weights is not None: losses *= weights
    losses = _average(losses, weights, average_across_timesteps, average_across_batch)  
    
    return losses  
