import pandas as pd
import numpy as np

credit_card_data = pd.read_csv('creditcard.csv')

#shuffle data to select a fraction of credit_card_data
shuffled_data = credit_card_data.sample(frac=1)
#take in the Class values and split it into one-hot data format, better way to feed in functions
# 0 -> [1 0]
# 1 -> [0 1]
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])
#normalize data to feed better into model
normalized_data = (one_hot_data - one_hot_data.min())/(one_hot_data.max() - one_hot_data.min())
#splitting X/y values
df_X = normalized_data.drop(['Class_0', 'Class_1'], axis=1)
df_y = normalized_data[['Class_0', 'Class_1']]
#convert data_frames to numpy arrays
ar_X, ar_y = np.asarray(df_X.values, dtype='float32'), np.asarray(df_y.values, dtype='float32')
#splitting data into train/test datasets
train_size = int(0.8 * len(ar_X))
(raw_X_train, raw_y_train) = (ar_X[:train_size], ar_y[:train_size])
(raw_X_test, raw_y_test) = (ar_X[train_size:], ar_y[train_size:])

#fraud ratio
count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraud_ratio = float(count_fraud/(count_fraud+count_legit))
print('Percent of fradulent transactions: ', fraud_ratio)

#to remove bias
weighting = 1 / fraud_ratio
raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

import tensorflow as tf

# 30 cells for i/p
input_dimensions = ar_X.shape[1]
# 2 cells for o/p
output_dimensions = ar_y.shape[1]
# 100 cells for 1st layer
num_layer_1_cells = 100
# 150 cells for 2nd layer
num_layer_2_cells = 150

# assign values at run time
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

#used as i/p to test model
X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

#1st Layer takes i/p and passes o/p to 2nd Layer
weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')
#Layer 2 takes i/p from 1st and passes to 3rd
weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name='weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name='biases_2')
#Layer 3 takes i/p from 2nd and o/ps [1 0]/[0 1] depending or fraud or legit
weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name='weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_3')

# creating a neural network
def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), 0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3

# Used to predict what results will be given during training or testing input data
y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

# calculates difference between actual o/p and predicted o/p
cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

# to minimize loss by changing the 3 layers' variable values
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])

num_epochs = 100

import time

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):

        start_time = time.time()

        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        if epoch % 10 == 0:
            timer = time.time() - start_time

            print('Epoch: {}'.format(epoch), 'Current loss: {0:.4f}'.format(cross_entropy_score),
                  'Elapsed time: {0:.2f} seconds'.format(timer))

            final_y_test = y_test_node.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
            print("Current accuracy: {0:.2f}%".format(final_accuracy))

    final_y_test = y_test_node.eval()
    final_y_test_prediction = y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print("Final accuracy: {0:.2f}%".format(final_accuracy))

final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
print('Final fraud specific accuracy: {0:.2f}%'.format(final_accuracy))