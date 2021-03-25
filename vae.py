

import pdb
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Add, Multiply, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.losses import mse, binary_crossentropy
from keras import backend as K


class variational_autoencoder():
	def __init__(self,dataset_name='mnist',architecture='mlp'):
		
		X_train = self.load_data(dataset_name)
		optimizer = 'adadelta'#Adam(0.0002, 0.5) #

		# image parameters
		self.epochs = 30000
		self.error_list = np.zeros((self.epochs,1))
		self.img_rows = X_train.shape[1]
		self.img_cols = X_train.shape[2]
		self.img_channels = X_train.shape[3]
		self.img_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
		self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
		self.z_dim = 10
		self.architecture = architecture
		self.dataset_name = dataset_name

		# Build and compile the discriminator
		self.vae = self.build_vae()
		self.vae.summary()

	def build_vae(self):

		n_pixels = self.img_rows*self.img_cols*self.img_channels

		#build encoder
		input_img = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
		input_img_flatten = Flatten()(input_img)
		z = Dense(512,name='z_intermediate')(input_img_flatten)
		z = LeakyReLU(alpha=0.2)(z)
		# mean and variance parameters
		z_mean = Dense(self.z_dim, name='z_mean')(z)
		z_log_var = Dense(self.z_dim, name='z_log_var')(z)

		#sample the latent vector
		z_rand = Lambda(self.sampling, output_shape=(self.z_dim,))([z_mean, z_log_var])
		#save the encoder
		self.encoder = Model(input_img, [z_mean, z_log_var, z_rand], name='encoder')

		#build decoder
		latent_inputs = Input(shape=(self.z_dim,), name='z_sampling')
		y = Dense(512)(latent_inputs)
		y = LeakyReLU(alpha=0.2)(y)
		output_img = Dense(self.img_rows*self.img_cols*self.img_channels, activation='sigmoid')(y)
		self.decoder = Model(latent_inputs, output_img, name='decoder')

		#build encoder + decoder (total model)
		output_img = self.decoder(self.encoder(input_img)[2])
		vae = Model(input_img, output_img, name='vae_mlp')

		#create the total model
		vae = Model(input_img,output_img)
		#define the loss
		vae_loss = self.vae_loss(input_img_flatten, output_img,z_mean,z_log_var)
		vae.add_loss(vae_loss)
		vae.compile(optimizer='adam')

		return vae

	def sampling(self,args):
		#Reparameterization trick
		z_mean, z_log_var = args
		batch_size = K.shape(z_mean)[0]
		# sample random vector of size (batch_size,z_dim)
		epsilon = K.random_normal(shape=(batch_size,self.z_dim))
		z_sigma = K.exp(0.5 * z_log_var)
		z_epsilon = Multiply()([z_sigma,epsilon])
		z_rand = Add()([z_mean,z_epsilon])
		return z_rand

	def vae_loss(self,x,y,z_mean,z_log_var):
		# BEGIN FILL IN CODE

		# reconstruction loss

		# reconstruction_loss = n_pixels * binary_crossentropy(input_img_flatten,output_img)
		reconstruction_loss = K.sum(K.binary_crossentropy(x, y), axis=-1)
		pdb.set_trace()
		
		# KL divergence
		#kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_div = K.square(z_mean) + K.exp(z_log_var) - z_log_var - 1
		kl_div = 0.5 * K.sum(kl_div, axis=-1)
		vae_loss = K.mean(reconstruction_loss + kl_div)

		# END FILL IN CODE
		return vae_loss

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

		for i in range(0,epochs):

			# ---------------------
			#  Train variational autoencoder
			# ---------------------

			# Select a random batch of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			curr_batch = X_train[idx,:,:,:]
			# Autoencoder training
			loss = self.vae.train_on_batch(curr_batch,None)

			# print the losses
			print("%d [Loss: %f]" % (i, loss))
			self.error_list[i] = loss

			# Save some random generated images and the models at every sample_interval iterations
			if (i % sample_interval == 0):
				n_images = 5
				idx = np.random.randint(0, X_train.shape[0], n_images)
				test_imgs = X_train[idx,:,:,:]
				self.reconstruct_images(test_imgs,'images/'+self.dataset_name+'_reconstruction_%06d.png' % i)
				self.sample_images('images/'+self.dataset_name+'_random_samples_%06d.png' % i)

	def reconstruct_images(self, test_imgs, image_filename):
		n_images = test_imgs.shape[0]
		#get output images
		output_imgs = np.reshape(self.vae.predict( test_imgs ),(n_images,self.img_rows,self.img_cols,self.img_channels))
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

	def sample_images(self, image_filename):

		n_images = 8
		#get output images
		z_sample = np.random.normal(0,1,(n_images,self.z_dim))
		r = 1
		c = n_images
		fig, axs = plt.subplots(r, c)
		for j in range(c):
			x_decoded = np.reshape(self.decoder.predict(z_sample) , (n_images,self.img_rows,self.img_cols,self.img_channels))
			#black and white images
			axs[j].imshow(x_decoded[j, :,:,0], cmap='gray')
			axs[j].axis('off')
		fig.savefig(image_filename)
		plt.close()


if __name__ == '__main__':

	#choose dataset
	dataset_name = 'mnist'#

	#create AE model
	architecture = 'mlp'#'convolutional'#
	vae = variational_autoencoder(dataset_name,architecture)#,
	is_training = 1

	if (is_training ==1):
		vae.train(epochs=vae.epochs, batch_size=64, sample_interval=100)
		plt.plot(vae.error_list[30:])
		plt.show()
	else:
		vae.test_images('images/test_images.png')