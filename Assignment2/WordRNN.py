import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
BATCH_SIZE = 128
no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x):

  word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return logits, word_list

def data_read_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values

  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)


  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))


  no_words = len(vocab_processor.vocabulary_)

  return x_train, y_train, x_test, y_test, no_words

def main():
  global n_words

  x_train, y_train, x_test, y_test, n_words = data_read_words()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  logits, word_list = rnn_model(x)


  # One hot the Y labels
  y_one_hot = tf.one_hot(y_, MAX_LABEL)

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
    train_classi_error = []
    test_classi_error = []

    for epoch in range(no_epochs):
      np.random.shuffle(idx)
      x_train = x_train[idx]
      y_train = y_train[idx]
      for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
        train_op.run(feed_dict={x: x_train[start:end], y_:y_train[start:end]})
      
      # Training Dataset evaluations
      train_acc.append(accuracy.eval(feed_dict={x: x_train, y_: y_train})) # Training Accuracy %
      loss.append(entropy.eval(feed_dict={x: x_train, y_: y_train})) # Training Cross Entropy Loss
      train_classi_error.append(classification_error.eval(feed_dict={x: x_train, y_: y_train})) # Training Classification Errors

      # Testing Dataset Evaluations
      test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test})) # Test Accuracy %
      test_classi_error.append(classification_error.eval(feed_dict={x: x_test, y_: y_test})) # Testing Classification Errors

      if epoch%1 == 0:
        print('Epoch: %d, Cross-Entropy: %g'%(epoch, loss[epoch]))
  
  

  plt.figure(1)
  plt.plot(range(no_epochs), train_acc)
  plt.title('Word RNN Classifier Train Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Train Accuracy')
  plt.savefig('./Snapshots/WordRNNTrainAccuracy')

  plt.figure(2)
  plt.plot(range(no_epochs), loss)
  plt.title('Word RNN Classifier Cross Entropy Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Cross-Entropy-Cost')
  plt.savefig('./Snapshots/WordRNNCrossEntropyLoss')


  plt.figure(3)
  plt.plot(range(no_epochs), train_classi_error)
  plt.title('Word RNN Classifier Train Classification Errors')
  plt.xlabel('Epochs')
  plt.ylabel('Train Classification Errors')
  plt.savefig('./Snapshots/WordRNNTrainingClassificationErrors')


  plt.figure(4)
  plt.plot(range(no_epochs), test_acc)
  plt.title('Word RNN Classifier Test Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Test Accuracy')
  plt.savefig('./Snapshots/WordRNNTestAccuracy')


  plt.figure(5)
  plt.plot(range(no_epochs), test_classi_error)
  plt.title('Word RNN Classifier Test Classification Errors')
  plt.xlabel('Epochs')
  plt.ylabel('Test Classification Errors')
  plt.savefig('./Snapshots/WordRNNTestClassificationErrors')



if __name__ == '__main__':
  main()
