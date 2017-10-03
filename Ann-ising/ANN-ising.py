import tensorflow as tf
import input_data
import sys


## Code from article "Machine Learning phases of matter" 


L=20
n_labels=2
hiddenunits1=400
reg=0.05


Ntemp=20
samples_per_T=250
Nord=20 

mnist = input_data.read_data_sets(n_labels, L, 'txt', one_hot=True)


train_subset = 10000


graph = tf.Graph()
with graph.as_default():
  x = tf.placeholder("float", shape=[None, L*L])
  y = tf.placeholder("float", shape=[None, n_labels])

  W1 = tf.Variable(tf.truncated_normal([L*L,hiddenunits1], stddev=0.01))
  b1 = tf.constant(0.01, shape=[hiddenunits1])
  Layer1 = tf.nn.sigmoid(tf.matmul(x, W1)+b1)

  W2 = tf.Variable(tf.truncated_normal([hiddenunits1,n_labels], stddev=0.01))
  b2 = tf.constant(0.01, shape=[n_labels])
  y_out = tf.nn.sigmoid(tf.matmul(Layer1, W2)+b2)

  cross_entropy = tf.reduce_sum(-y*tf.log(y_out) - (1.0 - y)*tf.log(1.0 - y_out)) + reg*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))
  optimizer= tf.train.AdamOptimizer(0.0001)
  train_op = optimizer.minimize(cross_entropy)

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_out,1), tf.argmax(y,1)), "float"))


with tf.Session(graph=graph) as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
      train_accuracy = sess.run(accuracy,feed_dict={x:batch[0], y: batch[1]})
      print("step %d, training accuracy %g" %(i, train_accuracy))
    sess.run(train_op, feed_dict={x: batch[0], y: batch[1]})

  print("test accuracy ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

  ii=0
  for i in range(Ntemp):
    av=0.0
    for j in range(samples_per_T):
      batch=(mnist.test.images[ii,:].reshape((1,L*L)),mnist.test.labels[ii,:].reshape((1,n_labels)))     
      res=sess.run(y_out,feed_dict={x: batch[0], y: batch[1]})
      av=av+res 
      ii=ii+1 
    av=av/samples_per_T
    print(i,av)   
       

# accuracy vs temperature
  for ii in range(Ntemp):
    batch=(mnist.test.images[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape(samples_per_T,L*L), mnist.test.labels[ii*samples_per_T:ii*samples_per_T+samples_per_T,:].reshape((samples_per_T,n_labels)) )
    train_accuracy = sess.run(accuracy,feed_dict={x:batch[0], y: batch[1]}) 
    print(ii, train_accuracy)


