import tensorflow as tf
import nltk
import numpy as np

def cross_entropy(train_logits,targets,target_lengths):
    eval_mask = tf.sequence_mask(target_lengths, dtype=tf.float32)
    xent_loss = tf.contrib.seq2seq.sequence_loss(train_logits, targets, eval_mask)
    return xent_loss

def min_affective_dissonance(param,train_logits,targets,target_lengths,enc_embed_input,dec_embed_input):
    lamba = tf.Variable(param, name="lamba")
    lamba_inv=tf.Variable(-(1.0-param),name="lamba_inv")
    predicted_prob = tf.reduce_max(dec_embed_input)
      
    affect_distance = tf.nn.l2_normalize(tf.subtract(tf.reduce_mean(enc_embed_input,1), tf.reduce_mean(dec_embed_input,1)))
    
    val =tf.reduce_sum((tf.multiply(lamba,tf.multiply(predicted_prob,affect_distance))))
  
    final_cost = tf.add(tf.multiply(lamba_inv,cross_entropy(train_logits,targets,target_lengths)),val)
    return final_cost

def max_affective_dissonance(param,train_logits,targets,target_lengths,enc_embed_input,dec_embed_input):
    lamba = tf.Variable(param, name="lamba")
    lamba_inv=tf.Variable(-(1.0-param),name="lamba_inv")
    predicted_prob = tf.reduce_max(dec_embed_input)
      
    affect_distance = tf.nn.l2_normalize(tf.subtract(tf.reduce_mean(enc_embed_input,1), tf.reduce_mean(dec_embed_input,1)))
    
    val =tf.reduce_sum((tf.multiply(lamba,tf.multiply(predicted_prob,affect_distance))))
  
    final_cost = tf.subtract(tf.multiply(lamba_inv,cross_entropy(train_logits,targets,target_lengths)),val)
    return final_cost

def max_affective_content(param,train_logits,targets,target_lengths,dec_embed_input):
    lamba = tf.Variable(param, name="lamba")
    lamba_inv=tf.Variable(-(1.0-param),name="lamba_inv")
    predicted_prob = tf.reduce_max(dec_embed_input)
    eta =[5,1,5]  
    affect_distance = tf.nn.l2_normalize(tf.subtract(dec_embed_input,eta))
    
    val =tf.reduce_sum((tf.multiply(lamba,tf.multiply(predicted_prob,affect_distance))))
  
    final_cost = tf.subtract(tf.multiply(lamba_inv,cross_entropy(train_logits,targets,target_lengths)),val)
    return final_cost