import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
EMBEDDING_SIZE = 20
FILTER_SHAPE1 = [20, EMBEDDING_SIZE] # First Conv Filter
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


def word_cnn_model(x):

	# Embedded Layer of size 20
	with tf.name_scope("embedding"):
		init_embeddings = tf.random_uniform([n_words, EMBEDDING_SIZE])
		embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
		word_vector = tf.nn.embedding_lookup(embeddings, x)
		word_vector = tf.expand_dims(word_vector, -1)

		print(word_vector)

	# word_vector = tf.reshape(tf.contrib.layers.embed_sequence(
 #      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE), [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1])

	with tf.variable_scope('CNN_Layer1'):
	    conv1 = tf.layers.conv2d(
	        inputs=word_vector,
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
	return word_vector, logits 


def read_data_words():
  
	x_train, y_train, x_test, y_test = [], [], [], []

	with open('train_medium.csv', encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
		  x_train.append(row[2])
		  y_train.append(int(row[0]))

	with open('test_medium.csv', encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
		  x_test.append(row[2])
		  y_test.append(int(row[0]))

	x_train = pandas.Series(x_train)
	y_train = pandas.Series(y_train)
	x_test = pandas.Series(x_test)
	y_test = pandas.Series(y_test)

	vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

	print(vocab_processor)

	x_train = np.array(list(vocab_processor.fit_transform(x_train)))
	x_test = np.array(list(vocab_processor.transform(x_test)))

	# print(x_train[0]) # Correct

	no_words = len(vocab_processor.vocabulary_)
	print('Total words: {}'.format(no_words))
	print(vocab_processor.vocabulary_._mapping)

	y_train = y_train.values
	y_test = y_test.values

	print("X train: {}".format(x_train[0]))
	print("Label train: {}".format(y_train[0]))
	print("X Test: {}".format(x_test[0]))
	print("Label Test: {}".format(y_test[0]))
	# print(y_test[0])

	return x_train, y_train, x_test, y_test, no_words

  
def main():
	global n_words
	x_train, y_train, x_test, y_test, n_words = read_data_words()

	print(len(x_train))
	print(len(x_test))

	# Create the model
	x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
	y_ = tf.placeholder(tf.int64)

	word_list_, logits = word_cnn_model(x)

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



		for epoch in range(no_epochs):
			np.random.shuffle(idx)
			x_train = x_train[idx]
			y_train = y_train[idx]
			for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
				train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
				# word_list_.run(feed_dict={x: x_train[start:end], y_:y_train[start:end]})

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
	plt.title('Word CNN Classifier Cross Entropy Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Cross-Entropy-Cost')
	plt.savefig('./Snapshots/WordCNNCrossEntropyLoss')

	plt.figure(2)
	plt.plot(range(no_epochs), test_acc)
	plt.title('Word CNN Classifier Test Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Test Accuracy')
	plt.savefig('./Snapshots/WordCNNCrossTestAccuracy')

	plt.figure(3)
	plt.plot(range(no_epochs), train_acc)
	plt.title('Word CNN Classifier Train Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Train Accuracy')
	plt.savefig('./Snapshots/WordCNNCrossTrainAccuracy')

	plt.figure(4)
	plt.plot(range(no_epochs), classi_error)
	plt.title('Word CNN Classifier Classification Errors')
	plt.xlabel('Epochs')
	plt.ylabel('Classification Errors')
	plt.savefig('./Snapshots/WordCNNCrossClassificationErrors')

if __name__ == '__main__':
	main()
