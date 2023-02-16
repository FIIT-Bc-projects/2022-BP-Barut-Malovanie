import os
import tensorflow as tf
import random

from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.ops.image_ops_impl import ResizeMethod


class DatasetInitializer:
    def __init__(self, regather=False):
        if regather:
            self.gather_descriptions('ex_2_flowers/data/labels_test')
        self.tokenizer = self.initialize_tokenizer()
        self.dict, self.embeddings_arr = self.create_embedding_dict()
        self.image_count = len(os.listdir('ex_2_flowers/data/flowers_test'))
        self.dataset = self.initialize_dataset()

    def load_dataset(self):
        return self.dataset

    def initialize_dataset(self):
        dataset = tf.data.Dataset.list_files('ex_2_flowers/data/flowers_test/*', shuffle=False)
        dataset = dataset.shuffle(self.image_count, reshuffle_each_iteration=True)
        dataset = dataset.take(self.image_count)
        dataset = dataset.map(lambda x: tf.py_function(self.process_path, [x], [tf.float32, tf.int32]),
                              num_parallel_calls=AUTOTUNE)
        return dataset

    @classmethod
    def gather_descriptions(cls, path_to_dir):
        path = Path(path_to_dir)
        master = open(os.path.join("all_descriptions.txt"), "w")
        for file in path.iterdir():
            with open(file, 'r')as f:
                for line in f:
                    master.write(line)
        master.close()

    @classmethod
    def process_text_input(cls, tokenizer, text, max_length):
        sequence = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(sequence, max_length)
        return padded

    @classmethod
    def initialize_tokenizer(cls, vocab_size=32_000, oov="<OOV>"):
        tokenizer = Tokenizer(vocab_size, oov_token=oov)
        with open('ex_2_flowers/data/all_descriptions.txt', 'r') as file:
            tokenizer.fit_on_texts(file.readlines())
        return tokenizer

    def create_embedding_dict(self, path_to_dir='ex_2_flowers/data/labels_test', max_length_seq=50):
        path = Path(path_to_dir)
        embedding_t = None
        names = []
        indexes = []
        for file in path.iterdir():
            with open(file, "r") as f:
                names.append(file.name[-15:-4]+'.jpg')
                indexes.append(int(file.name[-9:-4])-1)
                processed_t = tf.constant(self.process_text_input(self.tokenizer, f.readlines(), max_length_seq))
                processed_t = tf.reshape(processed_t, (1, 10, 50))
                if embedding_t is None:
                    embedding_t = processed_t
                else:
                    embedding_t = tf.concat([embedding_t, processed_t], 0)
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.constant(names), tf.constant(indexes)), default_value=-1), embedding_t

    def get_label(self, file_path):
        name = file_path[-15:]
        index = self.dict.lookup(tf.constant([name]))[0]
        label = random.choice(self.embeddings_arr[index])
        return label

    @classmethod
    def process_image(cls, img):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [64, 64], method=ResizeMethod.BICUBIC, antialias=True)

    def process_path(self, file_path):
        file_path = bytes.decode(file_path.numpy())
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.process_image(img)
        img = img / 255.0
        return img, label
