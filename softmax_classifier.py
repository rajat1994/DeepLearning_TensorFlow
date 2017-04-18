import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None,28,28,1])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))



#model

Y = tf.nn.softmax(tf.matmul(tf.reshape(X,[-1,784]),W)+b)

#placeholder for correct answers


Y_ = tf.placeholder(tf.float32, [None,10])

#loss function

cross_entropy = -tf.reduce_sum(Y_*tf.log(Y))

# % of correct answers found in batch

is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):

	batch_X, batch_Y = mnist.train.next_batch(100)
	train_data = {X:batch_X, Y_:batch_Y}

	sess.run(train_step, feed_dict =  train_data)

	a,c = sess.run([accuracy,cross_entropy], feed_dict = train_data)
	

	test_data = {X:mnist.test.images, Y_:mnist.test.labels}
	a,c = sess.run([accuracy, cross_entropy], feed_dict = test_data)
	print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))