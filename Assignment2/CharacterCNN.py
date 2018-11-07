import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256] # First Conv Filter
FILTER_SHAPE2 = [20, 1] # Second Conv Filter
POOLING_WINDOW = 4 # 4 x 4 Filter for Pooling
POOLING_STRIDE = 2 # Stride 2 > Start from 2 pixels
MAX_LABEL = 15	
BATCH_SIZE = 128

no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_cnn_model(x):
  
	input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1]) # 256 different characters / -1 means it will infer on the dimension

	print(input_layer.shape)

	with tf.variable_scope('CNN_Layer1'):
	    conv1 = tf.layers.conv2d(
	        inputs=input_layer,
	        filters=N_FILTERS,
	        kernel_size=FILTER_SHAPE1,
	        padding='VALID',
	        activation=tf.nn.relu)
	    print(conv1.shape)
	    pool1 = tf.layers.max_pooling2d(
	        inputs=conv1,
	        pool_size=POOLING_WINDOW,
	        strides=POOLING_STRIDE,
	        padding='SAME')
	    print(pool1.shape)
	with tf.variable_scope('CNN_Layer2'):
	    conv2 = tf.layers.conv2d(
	    	inputs=pool1,
	    	filters=N_FILTERS,
	    	kernel_size=FILTER_SHAPE2,
	    	padding='VALID',
	    	activation=tf.nn.relu)
	    print(conv2.shape)
	    pool2 = tf.layers.max_pooling2d(
	    	inputs=conv2,
	    	pool_size=POOLING_WINDOW,
	    	strides=POOLING_STRIDE,
	    	padding='SAME')
	    print(pool2.shape)

	    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])
	    print(pool2.shape)
	    
	logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)
	print(logits.shape)
	return input_layer, logits 


def read_data_chars():
  
	x_train, y_train, x_test, y_test = [], [], [], []

	with open('train_medium.csv', encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
		  x_train.append(row[1])
		  y_train.append(int(row[0]))

	with open('test_medium.csv', encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
		  x_test.append(row[1])
		  y_test.append(int(row[0]))

	x_train = pandas.Series(x_train)
	y_train = pandas.Series(y_train)
	x_test = pandas.Series(x_test)
	y_test = pandas.Series(y_test)


	char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
	x_train = np.array(list(char_processor.fit_transform(x_train)))
	x_test = np.array(list(char_processor.transform(x_test)))
	y_train = y_train.values
	y_test = y_test.values

	return x_train, y_train, x_test, y_test

  
def main():
  
	x_train, y_train, x_test, y_test = read_data_chars()

	print(len(x_train))
	print(len(x_test))

	# Create the model
	x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
	y_ = tf.placeholder(tf.int64)

	inputs, logits = char_cnn_model(x)

	# One hot the Y labels
	y_one_hot = tf.one_hot(y_, MAX_LABEL)

	# Optimizer
	entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=logits))
	train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

	
	# argmax returns the index of the largest value across axes of a tensor
	classification_error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(logits, axis=1), tf.argmax(y_one_hot, axis=1)), tf.int32))
	correct_prediction = tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_one_hot, axis=1)), tf.float32) # Cast to float
	accuracy = tf.reduce_mean(correct_prediction)

	

	N = len(x_train)
	idx = (np.arange(N))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# training
		loss = []
		test_acc = []
		train_acc = []
		classi_error = []
		for epoch in range(100):
			np.random.shuffle(idx)
			x_train = x_train[idx]
			y_train = y_train[idx]
			for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
				train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

			loss.append(entropy.eval(feed_dict={x: x_train, y_: y_train})) # Entropy Cost on Training Data
			train_acc.append(accuracy.eval(feed_dict={x: x_train, y_:y_train})) # Accuracy on Training Data
			test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test})) # Accuracy on Testing Data
			classi_error.append(classification_error.eval(feed_dict={x: x_train, y_: y_train}))

			if epoch%1 == 0:
				# print('Epochs: {}, Cross-Entropy Loss: {}'.format(epoch, loss[epoch]))
				print('Epochs: {}, Classification Errors: {}'.format(epoch, classi_error[epoch]))

	#plot learning curves
	plt.figure(1)
	plt.plot(range(no_epochs), loss)
	plt.title('Character CNN Classifier Cross Entropy Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Cross-Entropy-Cost')
	plt.savefig('./Snapshots/CharCNNCrossEntropyLoss')

	plt.figure(2)
	plt.plot(range(no_epochs), test_acc)
	plt.title('Character CNN Classifier Test Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Test Accuracy')
	plt.savefig('./Snapshots/CharCNNCrossTestAccuracy')

	plt.figure(3)
	plt.plot(range(no_epochs), train_acc)
	plt.title('Character CNN Classifier Train Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Train Accuracy')
	plt.savefig('./Snapshots/CharCNNCrossTrainAccuracy')

	plt.figure(4)
	plt.plot(range(no_epochs), classi_error)
	plt.title('Character CNN Classifier Classification Errors')
	plt.xlabel('Epochs')
	plt.ylabel('Classification Errors')
	plt.savefig('./Snapshots/CharCNNCrossClassificationErrors')

if __name__ == '__main__':
	# char_cnn_model()
	# read_data_chars()
	main()
