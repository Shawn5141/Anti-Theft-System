import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def LeakyReLU(x, leak=0.1, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1.0 + leak)
		f2 = 0.5 * (1.0 - leak)
		return f1 * x + f2 * abs(x)

def pad(tensor, num=1):
	return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")

def antipad(tensor, num=1):
	batch, h, w, c = tensor.shape.as_list()
	return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])


class Autoencoder:
	def __init__(self):
		# self.model =  ...
		self.x = tf.placeholder("float", [1, 128, 128, 3])
		self.count = 0
		self.encodeFeature = self.encoder()
		print("Build up an autoencoder.")
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.4
		self.sess = tf.Session(config = config)
		saver.restore(self.sess, tf.train.latest_checkpoint('./../model/similar_constrain_logs/'))
		print("Restore weights for autoencoder done.")

	# def encode(self, image):
	# 	# encode image
	# 	# for testing
	# 	if self.count==0:
	# 		self.count += 1
	# 		return np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
	# 	else:
	# 		return np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])

	def encoder(self, trainable=False):
		batch, height, width, _ = self.x.shape.as_list()

		with tf.variable_scope('autoencoder') as scope:
			concat_inputs = self.x
			with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
								trainable=trainable,
								weights_initializer=slim.variance_scaling_initializer(),
								activation_fn=LeakyReLU,
								padding='VALID'):

				weights_regularizer = slim.l2_regularizer(0.0004)
				with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
					conv0 = slim.conv2d(pad(concat_inputs), 64, 3, stride=2, scope='conv0') # 128-64
					conv1 = slim.max_pool2d(conv0, 2, scope='pool1') # 64-32
					conv2 = slim.conv2d(pad(conv1), 128, 3, stride=2, scope='conv2') #32-16
					conv3 = slim.max_pool2d(conv2, 2, scope='pool3') # 16-8
					conv4 = slim.conv2d(pad(conv3), 512, 3, stride=2, scope='conv4') # 8-4
					encode_conv4 = tf.reshape(conv4, [batch, 4*4*512])
					return encode_conv4
		