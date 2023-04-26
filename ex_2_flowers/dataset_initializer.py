import os

import tensorflow as tf

from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.ops.image_ops_impl import ResizeMethod


class DatasetInitializer:
    def __init__(self, batch_size, regather=False, test=False):
        if test:
            self.images_path = 'data/flowers_test'
            self.labels_path = 'data/labels_test'
        else:
            self.images_path = 'data/flowers'
            self.labels_path = 'data/labels'
        if regather:
            self.gather_descriptions(self.labels_path)
        self.tokenizer = self.initialize_tokenizer()
        self.embeddings_arr = self.create_embedding_arr()
        self.image_count = len(os.listdir(self.images_path))
        self.image_count = len(os.listdir(self.images_path))
        self.dataset = self.initialize_dataset(batch_size)

    def load_dataset(self):
        return self.dataset

    def initialize_dataset(self, batch_size):
        dataset = tf.data.Dataset.list_files(self.images_path+'/*/*', shuffle=False)
        #dataset = dataset.shuffle(self.image_count, reshuffle_each_iteration=True)
        #dataset = dataset.take(self.image_count)
        dataset = dataset.map(lambda x: tf.py_function(self.process_path, [x], [tf.float32, tf.int32]),
                              num_parallel_calls=AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda x, y: tf.py_function(self.choose_labels, [x, y], [tf.float32, tf.int32]),
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
        with open('data/all_descriptions.txt', 'r') as file:
            tokenizer.fit_on_texts(file.readlines())
        return tokenizer

    def create_embedding_arr(self, max_length_seq=20):
        with open('data/all_descriptions.txt', "r") as f:
            processed_t = tf.constant(self.process_text_input(self.tokenizer, f.readlines(), max_length_seq))
            return processed_t

    def choose_labels(self, img, labels):
        indexes = tf.random.uniform([len(labels)], 0, 10, dtype=tf.int32)
        indexes = 0
        label = tf.gather(self.embeddings_arr, labels+indexes)
        return img, label

    @classmethod
    def process_image(cls, img):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [64, 64], method=ResizeMethod.BICUBIC, antialias=True)

    def process_path(self, file_path):
        img = tf.io.read_file(file_path)
        index = int(bytes.decode(file_path.numpy())[-9:-4])
        index = (index - 1) * 10
        img = self.process_image(img)
        img = img / 255.0
        return img, index
