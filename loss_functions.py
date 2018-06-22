import tensorflow as tf
import nltk
import numpy as np

def cross_entropy(train_logits,targets,target_lengths):
    eval_mask = tf.sequence_mask(target_lengths, dtype=tf.float32)
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

def min_affective_dissonance(value,logits,enc_embed_input,full_embeddings,targets):
    param = tf.Variable(value, name="param")
    param_inv=tf.Variable((1.0-value),name="param_inv")
    softmaxed_logits = tf.nn.softmax(logits)
    predicted_prob = tf.reduce_max(softmaxed_logits,axis=-1)
    xent= -tf.log(predicted_prob)
    #xent = -tf.log(reduce_gather(softmaxed_logits,targets))
    input_embed= tf.expand_dims(tf.reduce_mean(enc_embed_input,axis=1), axis=1)
    predicted_ids = tf.argmax(softmaxed_logits,axis=-1)
    prediction_vectors = tf.nn.embedding_lookup(full_embeddings, predicted_ids)
    average_prediction_vectors = embed_predicted_timestep(prediction_vectors)
    affect_dissonance = predicted_prob*tf.norm(input_embed[:,:,1024:1027] - average_prediction_vectors[:,:,1024:1027])  
    loss = tf.reduce_mean((param_inv*xent) + (param*affect_dissonance))
    return loss

def max_affective_dissonance(value,logits,enc_embed_input,full_embeddings,targets):
    param = tf.Variable(value, name="param")
    param_inv=tf.Variable((1.0-value),name="param_inv")
    softmaxed_logits = tf.nn.softmax(logits)
    predicted_prob = tf.reduce_max(softmaxed_logits,axis=-1)
    xent= -tf.log(predicted_prob)
    #xent = -tf.log(reduce_gather(softmaxed_logits,targets))
    input_embed= tf.expand_dims(tf.reduce_mean(enc_embed_input,axis=1), axis=1)
    predicted_ids = tf.argmax(softmaxed_logits,axis=-1)
    prediction_vectors = tf.nn.embedding_lookup(full_embeddings, predicted_ids)
    average_prediction_vectors = embed_predicted_timestep(prediction_vectors)
    affect_dissonance = predicted_prob*tf.norm(input_embed[:,:,1024:1027] - average_prediction_vectors[:,:,1024:1027])    
    loss = tf.reduce_mean((param_inv*xent) - (param*affect_dissonance))
    return loss

def max_affective_content(value,logits,full_embeddings,targets):
    param = tf.Variable(value, name="param")
    param_inv=tf.Variable((1.0-value),name="param_inv")
    softmaxed_logits = tf.nn.softmax(logits)
    predicted_prob = tf.reduce_max(softmaxed_logits,axis=-1)
    xent= -tf.log(predicted_prob)
    #xent = -tf.log(reduce_gather(softmaxed_logits,targets))
    predicted_ids = tf.argmax(softmaxed_logits,axis=-1)
    prediction_vectors = tf.nn.embedding_lookup(full_embeddings, predicted_ids)
    neutral_vector = [5,1,5] 
    affect_dissonance = tf.norm(prediction_vectors[:,:,1024:1027] - neutral_vector)
    loss = tf.reduce_mean((param_inv*xent) - (param*predicted_prob*affect_dissonance))
    return loss  