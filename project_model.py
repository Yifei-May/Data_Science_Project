from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input, Conv2D, Reshape, Flatten, UpSampling2D, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#from google.colab import drive

# Load the dataset
def load_data():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = (x_train.astype(np.float32) - 127.5)/127.5
  
  # Convert shape from (60000, 28, 28) to (60000, 784)
  #x_train = x_train.reshape(60000, 784)
  return (x_train, y_train)

X_train, y_train = load_data()
print(X_train.shape, y_train.shape)

def build_discriminator_and_recognitor(img_shape=(128,128,1)):
    img = Input(shape=img_shape)
    model = Sequential()
    model.add(Conv2D(16, (3,3), strides=2, input_shape=img_shape))
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
    label = Dense(8, activation='tanh')(img_embedding)
    # Return discriminator and recognition network
    return Model(img, validity), Model(img, label)   

#按照论文自己调整之后的网络结构, 输出128*128*3
def build_generator_try(latent_dim=108):
    model = Sequential()
    model.add(Dense(6*6*256, input_dim=latent_dim)) #把这里的4改成了6
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
    model.add(Conv2DTranspose(3, (3,3), activation='tanh', strides=1))
    model.add(Dropout(0.6))
     
    gen_input = Input(shape=(latent_dim,))
    img = model(gen_input)
    print("The summary of generator:")
    model.summary()
    return Model(gen_input, img)

#按照论文写的网格结构, 输出18*18*3
def build_generator(latent_dim=108):
    model = Sequential()
    model.add(Dense(6*6*256, input_dim=latent_dim)) #把这里的4改成了6
    model.add(Reshape((6, 6, -1)))
    model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Conv2DTranspose(64, (3,3), activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Conv2DTranspose(32, (3,3), activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Conv2DTranspose(16, (3,3), activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Conv2DTranspose(16, (3,3), activation='relu', strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Conv2DTranspose(3, (3,3), activation='tanh', strides=1))
    model.add(Dropout(0.6))
     
    gen_input = Input(shape=(latent_dim,))
    img = model(gen_input)
    print("The summary of generator:")
    model.summary()
    
    return Model(gen_input, img)

#随便找了一个infoGAN的generator看了一下他的维度转换
#他的输入训练集照片的维度为28*28*1,最后generator输出也是28*28*1
def build_generator1(latent_dim = 72):
    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    gen_input = Input(shape=(latent_dim,))
    img = model(gen_input)

    model.summary()
    return Model(gen_input, img)

if __name__ == "__main__":
    discriminator,recognitor = build_discriminator_and_recognitor()
    generator_try=build_generator_try()
    generator = build_generator()
    generator1 = build_generator1()