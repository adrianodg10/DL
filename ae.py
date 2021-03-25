

import pdb
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


class autoencoder():
	def __init__(self,dataset_name='mnist',architecture='mlp'):
		
		X_train = self.load_data(dataset_name)
		optimizer = 'adadelta'#Adam(0.0002, 0.5) #

		# image parameters
		self.epochs = 5001
		self.error_list = np.zeros((self.epochs,1))
		self.img_rows = X_train.shape[1]
		self.img_cols = X_train.shape[2]
		self.img_channels = X_train.shape[3]
		self.img_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
		self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
		self.z_dim = 32
		self.architecture = architecture
		self.dataset_name = dataset_name

		# Build and compile the discriminator
		self.ae = self.build_ae()
		self.ae.summary()
		self.ae.compile(optimizer=optimizer, loss='binary_crossentropy') #binary cross-entropy loss, because mnist is grey-scale

	def build_ae(self):

		n_pixels = self.img_rows*self.img_cols*self.img_channels

		if (self.architecture == 'mlp'):
			# FULLY CONNECTED (MLP)

			#BEGIN FILL IN CODE
			#encoder
			input_img = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
			z = Flatten()(input_img)
			z = Dense(self.z_dim)(z)
			z = LeakyReLU(alpha=0.2)(z)
			# #DECODER
			output_img = Dense(784, activation='sigmoid')(z)
			output_img = Reshape(self.img_shape)(output_img)

			#END FILL IN CODE
		elif(self.architecture == 'convolutional'):
			# CONVOLUTIONAL MODEL

			#BEGIN FILL IN CODE
			input_img = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
			
			#encoder : 28x28x1 -> 14x14x8 -> 7x7x4 -> z_dim
			#28x28x1
			z = Conv2D(input_shape=self.img_shape,filters=8,kernel_size=(3,3),strides=(2,2),padding='same')(input_img)
			#14x14x16
			z = LeakyReLU(alpha=0.2)(z)
			z = Conv2D(filters=4,kernel_size=(3,3),strides=(2,2),padding='same')(z)
			#7x7x8
			z = LeakyReLU(alpha=0.2)(z)
			z = Flatten()(z)
			z = Dense(self.z_dim, input_dim=self.z_dim)(z)

			#DECODER : z_dim -> 7x7x4 -> 14x14x8 -> 28x28x1 ->
			y = Dense(7*7*4, input_dim=self.z_dim)(z)
			y = LeakyReLU(alpha=0.2)(y)
			y = Reshape((7,7,4))(y)
			y = Conv2DTranspose(filters=8,kernel_size=(4,4),strides=(2,2),padding='same')(y)
			# 14x14x16
			y = LeakyReLU(alpha=0.2)(y)
			y = Conv2DTranspose(filters=1,kernel_size=(4,4),strides=(2,2),padding='same')(y)
			# 28x28x1
			output_img = Activation('sigmoid')(y)

			#END FILL IN CODE

		#output the model
		return Model(input_img, output_img)


	def load_data(self,dataset_name):
		# Load the dataset
		if(dataset_name == 'mnist'):
			(X_train, _), (_, _) = mnist.load_data()
		else:
			print('Error, unknown database')

		# normalise images between 0 and 1
		X_train = X_train/255.0
		#add a channel dimension, if need be (for mnist data)
		if(X_train.ndim ==3):
			X_train = np.expand_dims(X_train, axis=3)
		return X_train

	def train(self, epochs, batch_size=128, sample_interval=50):
		
		#load dataset
		X_train = self.load_data(self.dataset_name)

		sigma = 0.0/255.0

		for i in range(0,epochs):

			# ---------------------
			#  Autoencoder
			# ---------------------

			# Select a random batch of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			curr_batch = X_train[idx,:,:,:]
			curr_batch_noisy = np.clip(curr_batch + sigma * np.random.normal(loc=0.0, scale=1.0, size=curr_batch.shape),0.,1.)
			# Autoencoder training
			loss = self.ae.train_on_batch(curr_batch,curr_batch_noisy)

			# print the losses
			print("%d [Loss: %f]" % (i, loss))
			self.error_list[i] = loss

			# Save some random generated images and the models at every sample_interval iterations
			if (i % sample_interval == 0):
				n_images = 5
				idx = np.random.randint(0, X_train.shape[0], n_images)
				test_imgs = X_train[idx,:,:,:]
				curr_batch_noisy = np.clip(test_imgs + sigma * np.random.normal(loc=0.0, scale=1.0, size=test_imgs.shape),0.,1.)
				self.test_images(curr_batch_noisy,'images/'+self.dataset_name+'_reconstruction_%06d.png' % i)

	def test_images(self, test_imgs, image_filename):
		n_images = test_imgs.shape[0]
		#get output imagesq
		output_imgs = self.ae.predict( test_imgs )
		
		r = 2
		c = n_images
		fig, axs = plt.subplots(r, c)
		for j in range(c):
			#black and white images
			axs[0,j].imshow(test_imgs[j, :,:,0], cmap='gray')
			axs[0,j].axis('off')
			axs[1,j].imshow(output_imgs[j, :,:,0], cmap='gray')
			axs[1,j].axis('off')
		fig.savefig(image_filename)
		plt.close()


if __name__ == '__main__':

	#create the output image directory
	if (os.path.isdir('images')==0):
		os.mkdir('images')

	#choose dataset
	dataset_name = 'mnist'#

	#create AE model
	architecture = 'mlp'#'convolutional'#
	ae = autoencoder(dataset_name,architecture)#,
	is_training = 1

	if (is_training ==1):
		ae.train(epochs=ae.epochs, batch_size=64, sample_interval=100)
		plt.plot(ae.error_list[30:])
		plt.show()
	else:
		ae.test_images('images/test_images.png')