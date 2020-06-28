from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input, Conv2D, Reshape, Flatten, UpSampling2D, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import to_categorical
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class INFOGAN:
    def __init__(self):
        self.img_shape = (128, 128, 1)
        self.num_classes = 8
        self.latent_dim = 100+8

        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        # Build the discriminator and recognitor network
        self.discriminator, self.recognitor = self.build_discriminator_and_recognitor()
        self.discriminator.compile(loss=['binary_crossentropy'],
                    optimizer=optimizer,
                    metrics=['accuracy'])
        self.recognitor.compile(loss=self.mutual_info_loss,
                    optimizer=optimizer,
                    metrics=['accuracy']) 

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated image as input and determines validity
        valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.recognitor(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=['binary_crossentropy', self.mutual_info_loss],
                            optimizer=optimizer)

    def build_discriminator_and_recognitor(self):

        img = Input(shape=self.img_shape)
        model = Sequential()
        model.add(Conv2D(16, (3,3), strides=2, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3,3), strides=1))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3,3), strides=2))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3,3), strides=1))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(256, (3,3), strides=2))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(16, (3,3), strides=1))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Flatten())
        print("The summary of the sharing part of discriminator and recognitor:")
        model.summary()

        img_embedding = model(img)
        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)
        # Recognition
        label = Dense(self.num_classes, activation='tanh')(img_embedding) #the num of label type
        # Return discriminator and recognition network
        return Model(img, validity), Model(img, label)  

    def build_generator(self):

        model = Sequential()
        model.add(Dense(6*6*256, input_dim=self.latent_dim)) #把这里的4改成了6
        model.add(Reshape((6, 6, -1)))
        model.add(Conv2DTranspose(128, (3,3), activation='relu', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Conv2DTranspose(64, (3,3), activation='relu', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Conv2DTranspose(32, (3,3), activation='relu', strides=1))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Conv2DTranspose(16, (3,3), activation='relu', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Conv2DTranspose(16, (3,3), activation='relu', strides=1))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))
        model.add(Conv2DTranspose(16, (3,3), activation='relu', strides=1))
        model.add(BatchNormalization())
        model.add(UpSampling2D())
        model.add(Dropout(0.6))
        model.add(Conv2DTranspose(1, (3,3), activation='tanh', strides=1))
        model.add(Dropout(0.6))
        
        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)
        print("The summary of generator:")
        model.summary()
        return Model(gen_input, img)

    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def sample_generator_input(self, size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (size, 100))
        sampled_labels = np.random.randint(0, self.num_classes, size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        return sampled_noise, sampled_labels

    def train(self, epochs, batch_size=128):

        # Load the dataset
        train_realDataGen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        train_realGenerator = train_realDataGen.flow_from_directory(
            os.getcwd()+"/data",
            target_size=(128,128),
            color_mode="grayscale", #单通道照片
            batch_size=batch_size)  
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select half batch of images randomly
            # ind = np.random(0, x_train.shape[0], batch_size)
            # imgs = x_train[ind]
            imgs, imgs_labels = train_realGenerator.next()

            # Sampled noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
            # Generate half batch of fake images
            gen_imgs = self.generator.predict(gen_input)

            # Train on real and generated data
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator and recognitor
            # ---------------------
            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %.2f, acc.: %.2f%%] [R loss: %.2f] [G loss: %.2f]" 
                % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))
            
            # If at save interval => save generated image samples
            if epoch % 20 == 0:
                self.sample_images(epoch)
            
    def sample_images(self, epoch):
        r, c = 10, 8

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(r)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5 #???
            for j in range(r):
                axs[j,i].imshow(gen_imgs[j,:,:,0], cmap='gray')
                axs[j,i].axis('off')
        fig.savefig(os.getcwd()+"/generated_images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = os.getcwd+"/saved_model/%s.json" % model_name
            weights_path = os.getcwd+"/saved_model/%s_weights.h5" % model_name

            json_string = model.to_json()
            open(model_path, 'w').write(json_string)
            print("the model architecture of %s saved to %s"%(model_name, model_path))
            model.save_weights(weights_path)
            print("the model weights of %s saved to %s"%(model_name, weights_path))

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == "__main__":
    infogan=INFOGAN()
    infogan.train(epochs=100, batch_size=128)
