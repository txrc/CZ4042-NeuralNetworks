import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt
import time

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256] # First Conv Filter
FILTER_SHAPE2 = [20, 1] # Second Conv Filter
POOLING_WINDOW = 4 # 4 x 4 Filter for Pooling
POOLING_STRIDE = 2 # Stride 2 > Start from 2 pixels
MAX_LABEL = 15	
BATCH_SIZE = 128

no_epochs = 5
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)


def char_cnn_model(x):
  
	input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1]) # 256 different characters / -1 means it will infer on the dimension


	with tf.variable_scope('CNN_Layer1'):
	    conv1 = tf.layers.conv2d(
	        inputs=input_layer,
	        filters=N_FILTERS,
	        kernel_size=FILTER_SHAPE1,
	        padding='VALID',
	        activation=tf.nn.relu)
      
	    pool1 = tf.layers.max_pooling2d(
	        inputs=conv1,
	        pool_size=POOLING_WINDOW,
	        strides=POOLING_STRIDE,
	        padding='SAME')
      
	with tf.variable_scope('CNN_Layer2'):
	    conv2 = tf.layers.conv2d(
	    	inputs=pool1,
	    	filters=N_FILTERS,
	    	kernel_size=FILTER_SHAPE2,
	    	padding='VALID',
	    	activation=tf.nn.relu)

	    pool2 = tf.layers.max_pooling2d(
	    	inputs=conv2,
	    	pool_size=POOLING_WINDOW,
	    	strides=POOLING_STRIDE,
	    	padding='SAME')


	    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

	    
	logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

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


	N = len(x_train)
	idx = np.arange(N)

	np.random.shuffle(idx)

	x_train, y_train = x_train[idx], y_train[idx]

	x_validation = x_train[5040:]
	y_validation = y_train[5040:]
	trainX = x_train[:5040]
	trainY = y_train[:5040]



	return trainX, trainY, x_validation, y_validation, x_test, y_test

  
def main():
  
	x_train, y_train, x_validation, y_validation, x_test, y_test = read_data_chars()

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
		time_to_run = 0
		train_loss = []
		validate_loss = []
		test_loss = []

		train_acc = []
		validate_acc = []
		test_acc = []

		train_classi_error = []
		validate_classi_error = []
		test_classi_error = []

		for epoch in range(no_epochs):
			np.random.shuffle(idx)
			x_train = x_train[idx]
			y_train = y_train[idx]
			t = time.time()
			for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
				train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
			time_to_run += time.time() - t

			# Validation Loss
			validate_loss.append(entropy.eval(feed_dict={x: x_validation, y_: y_validation}))


			# Entropy Cost on Training and Test Data
			train_loss.append(entropy.eval(feed_dict={x: x_train, y_: y_train})) 
			test_loss.append(entropy.eval(feed_dict={x: x_test, y_: y_test}))

			# Accuracy on Training and Test Data
			train_acc.append(accuracy.eval(feed_dict={x: x_train, y_:y_train})) # Accuracy on Training Data
			test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test})) # Accuracy on Testing Data

			# Classification Errors on Training and Test Data
			train_classi_error.append(classification_error.eval(feed_dict={x: x_train, y_: y_train}))
			test_classi_error.append(classification_error.eval(feed_dict={x: x_test, y_: y_test}))

			if epoch%1 == 0:
				# print('Epochs: {}, Cross-Entropy Loss: {}'.format(epoch, loss[epoch]))
				print('Epochs: {}, Training Loss: {}, Validation Loss: {}'.format(epoch, train_loss[epoch], validate_loss[epoch]))
		print("Total Time to Train the Network is: {}".format(time_to_run))
	#plot learning curves


	plt.figure(1)
	plt.plot(range(no_epochs), train_loss, 'g', label="Training Loss")
	plt.plot(range(no_epochs), validate_loss, 'b', label="Validation Loss")
	plt.title('Char CNN Train Loss vs Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Cross-Entropy-Loss')
	plt.legend()
	plt.savefig('./CharCNNTrainLossVsValidationLoss-EarlyStopping')

	plt.figure(2)
	plt.plot(range(no_epochs), train_loss)
	plt.title('Char CNN Classifier Training Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Train Cross-Entropy-Cost')
	plt.savefig('./CharCNNTrainCrossEntropyLoss-EarlyStopping')


	plt.figure(3)
	plt.plot(range(no_epochs), test_loss)
	plt.title('Char CNN Classifier Testing Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Test Cross-Entropy-Cost')
	plt.savefig('./CharCNNTestCrossEntropyLoss-EarlyStopping')

	plt.figure(4)
	plt.plot(range(no_epochs), train_acc)
	plt.title('Char CNN Classifier Train Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Train Accuracy')
	plt.savefig('./CharCNNTrainAccuracy-EarlyStopping')

	plt.figure(5)
	plt.plot(range(no_epochs), test_acc)
	plt.title('Char CNN Classifier Test Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Test Accuracy')
	plt.savefig('./CharCNNTestAccuracy-EarlyStopping')

	plt.figure(6)
	plt.plot(range(no_epochs), train_classi_error)
	plt.title('Char CNN Classifier Train Classification Errors')
	plt.xlabel('Epochs')
	plt.ylabel('Train Classification Errors')
	plt.savefig('./CharCNNTrainClassificationErrors-EarlyStopping')

	plt.figure(7)
	plt.plot(range(no_epochs), test_classi_error)
	plt.title('Char CNN Classifier Test Classification Errors')
	plt.xlabel('Epochs')
	plt.ylabel('Test Classification Errors')
	plt.savefig('./CharCNNTestClassificationErrors-EarlyStopping')

if __name__ == '__main__':
	# char_cnn_model()
	# read_data_chars()
	main()