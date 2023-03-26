import datetime

from ex_2_flowers.dataset_initializer import DatasetInitializer
from ex_2_flowers.models import Models
from utils import *

init = DatasetInitializer(64)
dataset = init.load_dataset()
models = Models()
generator = models.generator_model
discriminator = models.discriminator_model
gan = models.gan_model

def train_gan(generator, disc, gan, dataset, latent_dim, epochs=600, batch_size=256):
    dataset_size = 8189
    batch_in_epoch = int(dataset_size / batch_size)
    half_batch = int(batch_size / 2)
    acc_real_arr = []
    acc_fake_arr = []
    d_loss_arr = []
    g_loss_arr = []
    visualization_prompts = ['this flower has a ghostly lavender petals surrounding curly green stamen.',
                             'thin white anther are surrounded by dark purple pointy petals.',
                             'the flower has petals that are mostly red and pink and has petals that are soft, smooth and forming a disc like shape',
                             'the one petaled white flower has a yellow tip and fully leafed stem.',
                             'the flower has smooth white petals with yellow stamen and a green pollen tube',
                             'this flower has six fan shaped white petals surrounding the yellow stamen.',
                             'the flower has petals that are burgundy, drooping with white spots.',
                             'the flower has yellow petals that surround the black stamen',
                             'this orange flower has a lot of fruit like stamen',
                             'this flower has long, very thin red petals with ruffled yellow edges.']
    vis_embed = init.process_text_input(init.tokenizer, visualization_prompts, 20)

    for i in range(epochs):
        j = 0
        print('>%d' % (i+1))
        if (i+1) % 50 == 0:
            gen_path = 'saved/generator_model_%03d.h5' % (i+1)
            images_path = 'fake_flowers_%03d.png' % (i+1)
            fakeX, _ = generate_con_fake_samples(generator, vis_embed, 128, 10)
            visualize_images(fakeX, (1, 10), images_path)
            models.save_gen(gen_path)

        for realX, embed in dataset:
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

begin = datetime.datetime.now()
metrics = train_gan(generator, discriminator, gan, dataset, 128, 600, 64)
end = datetime.datetime.now()
print(f'Training took: {end-begin} seconds')


pyplot.plot(metrics[0], label='accuracy real')
pyplot.plot(metrics[1], label='accuracy fake')
pyplot.plot([0.5 for x in range(len(metrics[0]))], label='ideal accuracy')
pyplot.xlabel("Epoch")
pyplot.ylabel("Accuracy")
pyplot.legend()
pyplot.savefig('saved/acc_plot')
pyplot.close()

pyplot.plot(metrics[2], label='discriminator')
pyplot.plot(metrics[3], label='generator')
pyplot.xlabel("Epoch")
pyplot.ylabel("Loss")
pyplot.legend()
pyplot.savefig('saved/loss_plot')
pyplot.close()