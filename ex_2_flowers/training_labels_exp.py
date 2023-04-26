import datetime
import os
import numpy

from ex_2_flowers.dataset_init_labels_exp import DatasetInitializer
from ex_2_flowers.models import Models
from utils import *

#init = DatasetInitializer('data/flowers_label_exp/', 10)
##dataset = init.load_dataset()
#for x,y in dataset:
    #print(x.shape)

models = Models()
generator = models.generator_model
discriminator = models.discriminator_model
gan = models.gan_model

def train_gan(generator, disc, gan, latent_dim, epochs=600, batch_size=256):
    init = DatasetInitializer('data/flowers_label_exp/', batch_size)
    dataset = init.load_dataset()
    acc_real_arr = []
    acc_fake_arr = []
    d_loss_arr = []
    g_loss_arr = []
    visualization_prompts = ['this is flwoer one', 'this is flwoer one', 'this is flwoer one', 'this is flwoer one', 'this is flwoer one',
                             'this is flower two', 'this is flower two', 'this is flower two', 'this is flower two', 'this is flower two']
    vis_embed = init.process_text_input(init.tokenizer, visualization_prompts, 4)

    for i in range(epochs):
        j = 0
        print('>>>>>>%d' % (i+1))
        if (i+1) % 50 == 0:
            gen_path = 'saved/generator_model_%03d.h5' % (i+1)
            images_path = 'fake_flowers_%03d.png' % (i+1)
            fakeX, _ = generate_con_fake_samples(generator, vis_embed, 128, 10)
            visualize_images(fakeX, (1, 10), save=True, save_path=images_path)
            models.save_gen(gen_path)

        for realX, embed in dataset:
            print('==%d' % (j+1))
            size = realX.shape[0]
            realy = np.ones((size, 1))
            fakeX, fakey = generate_con_fake_samples(generator, embed, 128, size)
            d_loss_real = disc.train_on_batch(realX, realy)
            d_loss_fake = disc.train_on_batch(fakeX, fakey)
            d_loss = (d_loss_fake + d_loss_real) / 2
            latent_vectors = generate_latent_points(latent_dim, size)
            y_gan = np.ones((size, 1))
            g_loss = gan.train_on_batch([embed, latent_vectors], y_gan)
            if j == 0:
                acc_real = disc.evaluate(realX, realy, verbose=0)
                acc_fake = disc.evaluate(fakeX, fakey, verbose=0)
                acc_real_arr.append(acc_real)
                acc_fake_arr.append(acc_fake)
                d_loss_arr.append(d_loss)
                g_loss_arr.append(g_loss)
            j += 1

    return acc_real_arr, acc_fake_arr, d_loss_arr, g_loss_arr

batch_size = 1
begin = datetime.datetime.now()
metrics = train_gan(generator, discriminator, gan, 128, 600, batch_size)
end = datetime.datetime.now()
pyplot.close()
print(f'Training took: {end-begin} seconds')
save_acc_plot(metrics[0], metrics[1], 'acc_plot')
save_loss_plot(metrics[2], metrics[1], 'loss_plot')
