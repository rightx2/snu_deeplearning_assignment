
# coding: utf-8

# # M2177.003100 Deep Learning <br> Assignment #1 Part 3: Playing with Neural Networks by TensorFlow

# Copyright (C) Data Science Laboratory, Seoul National University. This material is for educational uses only. Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. Written by Jaehee Jang, September 2017

# Previously in `Assignment2-1_Data_Curation.ipynb`, we created a pickle with formatted datasets for training, development and testing on the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
#
# The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow.
#
# **Note**: certain details are missing or ambiguous on purpose, in order to test your knowledge on the related materials. However, if you really feel that something essential is missing and cannot proceed to the next step, then contact the teaching staff with clear description of your problem.
#
# ### Submitting your work:
# <font color=red>**DO NOT clear the final outputs**</font> so that TAs can grade both your code and results.
# Once you have done **part 1 - 3**, run the *CollectSubmission.sh* script with your **Student number** as input argument. <br>
# This will produce a compressed file called *[Your student number].tar.gz*. Please submit this file on ETL. &nbsp;&nbsp; (Usage: ./*CollectSubmission.sh* &nbsp; 20\*\*-\*\*\*\*\*)

# ## Load datasets
#
# First reload the data we generated in `Assignment2-1_Data_Curation.ipynb`.

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# In[2]:


pickle_file = 'data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[3]:


image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)

    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# ## TensorFlow tutorial: Fully Connected Network
#
# We're first going to train a **fully connected network** with *1 hidden layer* with *1024 units* using stochastic gradient descent (SGD).
#
# TensorFlow works like this:
# * First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. This description is all contained within the block below:
#
#       with graph.as_default():
#           ...
#
# * Then you can run the operations on this graph as many times as you want by calling `session.run()`, providing it outputs to fetch from the graph that get returned. This runtime operation is all contained in the block below:
#
#       with tf.Session(graph=graph) as session:
#           ...
#
# Let's load all the data into TensorFlow and build the computation graph corresponding to our training:

# In[4]:


# batch_size = 128
# nn_hidden = 1024

# graph = tf.Graph()
# with graph.as_default():
#     # Input data. For the training data, we use a placeholder that will be fed
#     # at run time with a training minibatch.
#     tf_train_dataset = tf.placeholder(
#         tf.float32,
#         shape=(batch_size, image_size * image_size)
#     )
#     tf_train_labels = tf.placeholder(
#         tf.float32,
#         shape=(batch_size, num_labels)
#     )
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)

#     # Variables.
#     w1 = tf.Variable(tf.truncated_normal([image_size * image_size, nn_hidden]))
#     b1 = tf.Variable(tf.zeros([nn_hidden]))
#     w2 = tf.Variable(tf.truncated_normal([nn_hidden, num_labels]))
#     b2 = tf.Variable(tf.zeros([num_labels]))

#     # Training computation.
#     hidden = tf.tanh(tf.matmul(tf_train_dataset, w1) + b1)
#     logits = tf.nn.softmax(tf.matmul(hidden, w2) + b2)

#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
#     )

#     # Optimizer.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)

#     valid_hidden = tf.tanh(tf.matmul(tf_valid_dataset, w1) + b1)
#     valid_prediction = tf.nn.softmax(tf.matmul(valid_hidden, w2) + b2)

#     test_hidden = tf.tanh(tf.matmul(tf_test_dataset, w1) + b1)
#     test_prediction = tf.nn.softmax(tf.matmul(test_hidden, w2) + b2)


# # Let's run this computation and iterate:

# # In[5]:


# num_steps = 10000

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     print("Initialized")
#     for step in range(num_steps):
#         # Pick an offset within the training data, which has been randomized.
#         # Note: we could use better randomization across epochs.
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#         # Generate a minibatch.
#         batch_data = train_dataset[offset:(offset + batch_size), :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#         # Prepare a dictionary telling the session where to feed the minibatch.
#         # The key of the dictionary is the placeholder node of the graph to be fed,
#         # and the value is the numpy array to feed to it.
#         _, l, predictions = session.run(
#             [
#                 optimizer,
#                 loss,
#                 train_prediction
#             ],
#             feed_dict={
#                 tf_train_dataset: batch_data,
#                 tf_train_labels: batch_labels
#             }
#         )
#         if (step % 1000 == 0):
#             print("Minibatch loss at step %d: %f" % (step, l))
#             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#             print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
#     print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
#     saver = tf.train.Saver()
#     saver.save(session, "./model_checkpoints/my_model_final")



# ---
# Problem
# -------
#
# Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.kr/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
#
# 1. Experiment with different hyperparameters: num_steps, learning rate, etc.
# 2. We used a fixed learning rate $\epsilon$ for gradient descent. Implement an annealing schedule for the gradient descent learning rate ([more info](http://cs231n.github.io/neural-networks-3/#anneal)). *Hint*. Try using `tf.train.exponential_decay`.
# 3. We used a $\tanh$ activation function for our hidden layer. Experiment with other activation functions included in TensorFlow.
# 4. Extend the network to multiple hidden layers. Experiment with the layer sizes. Adding another hidden layer means you will need to adjust the code.
# 5. Introduce and tune regularization method (e.g. L2 regularization) for your model. Remeber that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should imporve your validation / test accuracy.
# 6. Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
#
# **Evaluation:** Rank by test accuracy. If we have ties, we will test with another dataset which is not included in notMnist dataset.
#
# ---

# In[6]:
condition_dict = [
    # 96.2 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for batch size
    # 96.3 {"batch_size":256, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.2 {"batch_size":200, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 95.8 {"batch_size":64, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for fc3 hidden node size
    # 96.1 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":512, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.1 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":1024, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.3 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":2048, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for same hidden size
    # 96.3 {"batch_size":128, "layer1_size":4096, "layer2_size":4096, "layer3_size":4096, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.0 {"batch_size":128, "layer1_size":2048, "layer2_size":2048, "layer3_size":2048, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.0 {"batch_size":128, "layer1_size":1024, "layer2_size":1024, "layer3_size":1024, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for beta
    # 96.5 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 92.6 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.01, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 82.8 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.1, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for stddev
    # 96.1 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":1, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.3 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":3, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for dropout
    # 95.4 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":True, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # for learning rate
    # 92.8 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.01, "decay_steps":1000, "decay_rate":0.7, "num_steps": 100000},
    # 95.5 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.1, "decay_steps":1000, "decay_rate":0.7, "num_steps": 50000},
    # 96.0{"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.3, "decay_steps":1000, "decay_rate":0.7, "num_steps": 20000},
    # 10 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.9, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},



    # # decay step
    # 95.8 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":2000, "decay_rate":0.7, "num_steps": 12000},
    # 96.4 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":2000, "decay_rate":0.7, "num_steps": 24000},
    # 95.0 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":4000, "decay_rate":0.7, "num_steps": 12000},
    # 96.5 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":4000, "decay_rate":0.7, "num_steps": 50000},

    # # decay rate
    # 95.2 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.3, "num_steps": 12000},
    # 95.7 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.5, "num_steps": 12000},
    # 95.4 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001438, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.9, "num_steps": 12000},


    # # for beta의 upgrade version
    # # beta 변경
    # 96.7
    {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0005, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.5 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0003, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.6 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0001, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # stddev 변경
    # 10 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001, "stddev":3, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},
    # 96.2{"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001, "stddev":1, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # layer3_size 변경 => 더 낮아져버렸네
    # 96.4 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":2048, "beta":0.001, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},

    # # dropout 추가
    # 95.7 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.001, "stddev":2, "dropout":True, "learning_rate":0.5, "decay_steps":1000, "decay_rate":0.7, "num_steps": 12000},


    # More upgrade
    # 96.6 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0005, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":4000, "decay_rate":0.7, "num_steps": 50000},
    # 96.8
    {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0003, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":4000, "decay_rate":0.7, "num_steps": 50000},
    # 96.7
    {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0001, "stddev":2, "dropout":False, "learning_rate":0.5, "decay_steps":4000, "decay_rate":0.7, "num_steps": 50000},

    # 위에것(96.8)에서 더더욱 upgrade
    # learning rate
    # 96.8
    {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0003, "stddev":2, "dropout":False, "learning_rate":0.3, "decay_steps":4000, "decay_rate":0.7, "num_steps": 50000},
    # decay rate
    # 96.6 {"batch_size":128, "layer1_size":4096, "layer2_size":2048, "layer3_size":128, "beta":0.0003, "stddev":2, "dropout":False, "learning_rate":0.3, "decay_steps":4000, "decay_rate":0.5, "num_steps": 50000},
]

for condition in condition_dict:
    batch_size = condition["batch_size"]
    layer1_size = condition["layer1_size"]
    layer2_size = condition["layer2_size"]
    layer3_size = condition["layer3_size"]

    graph = tf.Graph()
    with graph.as_default():
        train_x = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        train_y = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        valid_set = tf.constant(valid_dataset)
        test_set = tf.constant(test_dataset)

        reg_param = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)  # count the number of steps taken.

        W1 = tf.Variable(tf.truncated_normal([image_size * image_size, layer1_size], stddev=np.sqrt(condition["stddev"] / (image_size * image_size))))
        b1 = tf.Variable(tf.zeros([layer1_size]))

        W2 = tf.Variable(tf.truncated_normal([layer1_size, layer2_size], stddev=np.sqrt(condition["stddev"] / layer1_size)))
        b2 = tf.Variable(tf.zeros([layer2_size]))

        W3 = tf.Variable(tf.truncated_normal([layer2_size, layer3_size], stddev=np.sqrt(condition["stddev"] / layer2_size)))
        b3 = tf.Variable(tf.zeros([layer3_size]))

        W4 = tf.Variable(tf.truncated_normal([layer3_size, num_labels], stddev=np.sqrt(condition["stddev"] / layer3_size)))
        b4 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        y1 = tf.nn.relu(tf.matmul(train_x, W1) + b1)
        if condition["dropout"]:
            y1 = tf.nn.dropout(y1, 0.5)

        y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
        if condition["dropout"]:
            y2 = tf.nn.dropout(y2, 0.5)

        y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
        if condition["dropout"]:
            y3 = tf.nn.dropout(y3, 0.5)

        logits = tf.matmul(y3, W4) + b4

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_y)
        )

        loss = loss + reg_param * (
            tf.nn.l2_loss(W1)+
            tf.nn.l2_loss(b1)+
            tf.nn.l2_loss(W2)+
            tf.nn.l2_loss(b2)+
            tf.nn.l2_loss(W3)+
            tf.nn.l2_loss(b3)+
            tf.nn.l2_loss(W4)+
            tf.nn.l2_loss(b4)
        )

        # Optimizer
        learning_rate = tf.train.exponential_decay(
            condition["learning_rate"],
            global_step,
            condition["decay_steps"],
            condition["decay_rate"],
            staircase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)

        a1_valid = tf.nn.relu(tf.matmul(valid_set, W1) + b1)
        a2_valid = tf.nn.relu(tf.matmul(a1_valid, W2) + b2)
        a3_valid = tf.nn.relu(tf.matmul(a2_valid, W3) + b3)
        valid_logits = tf.matmul(a3_valid, W4) + b4
        valid_prediction = tf.nn.softmax(valid_logits)

        a1_test = tf.nn.relu(tf.matmul(test_set, W1) + b1)
        a2_test = tf.nn.relu(tf.matmul(a1_test, W2) + b2)
        a3_test = tf.nn.relu(tf.matmul(a2_test, W3) + b3)
        test_logits = tf.matmul(a3_test, W4) + b4
        test_prediction = tf.nn.softmax(test_logits)

    num_steps = condition["num_steps"]
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(graph=graph, config=config) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps + 1):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            _, l, predictions = session.run(
                [optimizer, loss, train_prediction],
                feed_dict={
                    train_x: batch_data,
                    train_y: batch_labels,
                    reg_param: condition["beta"],
                }
            )
            if (step % 3000 == 0):
                print(step)
                print("   Minibatch loss : {}".format(l))
                print("   Minibatch accuracy : {}".format(accuracy(predictions, batch_labels)))
                print("   Validation accuracy : {}".format(accuracy(valid_prediction.eval(), valid_labels)))
        print("Test accuracy: {}".format(accuracy(test_prediction.eval(), test_labels)))
    print("")
    print("")
    print("")
    print("")
    saver = tf.train.Saver()
    saver.save(session, "./model_checkpoints/my_model_final")