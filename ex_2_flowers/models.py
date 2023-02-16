from keras import Sequential, Input
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Embedding, Reshape, Concatenate, Conv2DTranspose, \
    Activation
from keras.optimizers import Adam
from keras.models import Model


class Models:
    def __init__(self):
        pass

    @property
    def generator(self):
        input_text = Input(shape=(50,))
        embed = Embedding(32_000, 50, input_length=50)(input_text)
        embed = Dense(8*8)(embed)
        embed = Reshape((8, 8, 3, 1))

        input_latent = Input(shape=(128,))
        latent = Dense(473*8*8*3)(input_latent)
        latent = Reshape((8, 8, 3, 437))(latent)

        gen = Concatenate()([embed, latent])
        gen = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)

        gen = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)

        gen = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        out = Activation('tahn')(gen)
        model = Model([embed, latent], out)
        return model

    @property
    def discriminator(self):
        input = Input(shape=(64, 64, 3))
        disc = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(input)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = Dropout(0.4)(disc)

        disc = Conv2D(128, (4, 4), padding='same')(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = Dropout(0.4)(disc)

        disc = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = Dropout(0.4)(disc)

        disc = Conv2D(256, (4, 4), padding='same')(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = Dropout(0.4)(disc)

        #disc = Conv2D(256, (2, 2), strides=(2, 2), padding='same')(disc)
        #disc = LeakyReLU(alpha=0.2)(disc)
        #disc = Dropout(0.4)(disc)

        #disc = Conv2D(512, (4, 4), padding='same')(disc)
        #disc = LeakyReLU(alpha=0.2)(disc)
        #disc = Dropout(0.4)(disc)

        disc = Flatten()(disc)
        out = Dense(1, activation='sigmoid')(disc)
        model = Model(input, out)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model

    @classmethod
    def gan(cls, gen, disc):
        for layer in disc.layers:
            layer.trainable = False
        gan_output = disc(gen.output)
        model = Model(gen.input, gan_output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        return model

