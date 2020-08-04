import tensorflow as tf

def classifier_token(bert_output):
    return bert_output[:, 0]

def average_tokens(bert_output):
    return tf.reduce_mean(bert_output, axis = 1)

def both(bert_output):
    x1 = classifier_token(bert_output)
    x2 = average_tokens(bert_output)
    return tf.concat([x1, x2], axis = 1)

str_to_func = {'cls' : classifier_token, 'avg' : average_tokens, 'both' : both, 'full' : None}
