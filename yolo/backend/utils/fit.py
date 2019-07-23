# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         saved_weights_name = 'best_weights.h5'):
    """A function that performs training on a general keras model.

    # Args
        model : keras.models.Model instance
        loss_func : function
            refer to https://keras.io/losses/

        train_batch_gen : keras.utils.Sequence instance
        valid_batch_gen : keras.utils.Sequence instance
        learning_rate : float
        saved_weights_name : str
    """
    # 1. create optimizer
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # 2. create loss function
    model.compile(loss=loss_func,
                  optimizer=optimizer)

    # 4. training
    train_start = time.time()
    try:
        model.fit_generator(generator = train_batch_gen,
                        steps_per_epoch  = len(train_batch_gen), 
                        epochs           = nb_epoch,
                        validation_data  = valid_batch_gen,
                        validation_steps = len(valid_batch_gen),
                        callbacks        = _create_callbacks(saved_weights_name),                        
                        verbose          = 1,
                        workers          = 3,
                        max_queue_size   = 8)
    except KeyboardInterrupt:
        save_tflite(model)
        raise

    _print_time(time.time()-train_start)
    save_tflite(model)
def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))

def save_tflite(model):
        output_node_names = [node.op.name for node in model.outputs]
        input_node_names = [node.op.name for node in model.inputs]
        sess = K.get_session()
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
        for n in constant_graph.node:
            print( n.name )
        graph_io.write_graph(constant_graph, "" , "model.pb", as_text=False)
        print(output_node_names)
        print(input_node_names)
        model.save("model.h5")

def _create_callbacks(saved_weights_name):
    # Make a few callbacks
    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.001, 
                       patience=10, 
                       mode='min', 
                       verbose=1)
    checkpoint = ModelCheckpoint(saved_weights_name, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)
    callbacks = [early_stop, checkpoint]
    return callbacks
