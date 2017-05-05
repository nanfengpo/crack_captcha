import tensorflow as tf
import cv2
from src.gen_model import create_layer

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string("input","","input file path")

keep_prob=tf.placeholder(tf.float32)
X=cv2.imread(FLAGS.input)
Y=create_layer(X,keep_prob)

with tf.Session() as sess:
    saver=tf.train.Saver()
    saver.restore(sess,"break.model")
    y=sess.run(Y,feed_dict={keep_prob:0.8})
    print(y)