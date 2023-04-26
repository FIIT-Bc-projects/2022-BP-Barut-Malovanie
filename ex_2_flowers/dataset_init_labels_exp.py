import os

import tensorflow as tf
import numpy as np

from pathlib import Path
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.ops.image_ops_impl import ResizeMethod


class DatasetInitializer:
    def __init__(self, images_path, batch_size):
        self.tokenizer = self.initialize_tokenizer()
        self.embeddings_arr = self.create_embedding_arr(images_path)
        self.image_count = self.count_images(images_path)
        self.dataset = self.initialize_dataset(images_path, batch_size)
        self.label_index_dict = self.create_label_index_dict(images_path)

    @classmethod
    def count_images(cls, root_images_path):
        dirs = os.listdir(root_images_path)
        count = 0
        for directory in dirs:
            count += len(os.listdir(root_images_path + directory))
        return count

    def load_dataset(self):
        return self.dataset

    def initialize_dataset(self, images_path,  batch_size):
        dataset = tf.data.Dataset.list_files(images_path + '*/*', shuffle=False)
        dataset = dataset.shuffle(self.image_count, reshuffle_each_iteration=True)
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

    def create_embedding_arr(self, images_path, max_length_seq=4):
        prompt = os.listdir(images_path)
        prompt = np.char.add('this is flower ', prompt)
        processed_t = tf.constant(self.process_text_input(self.tokenizer, prompt, max_length_seq))
        return processed_t

    @classmethod
    def create_label_index_dict(cls, images_path):
        labels = os.listdir(images_path)
        ret_dict = {}
        for i, label in enumerate(labels):
            ret_dict[label] = i
        return ret_dict

    def choose_labels(self, img, labels):
        label = tf.gather(self.embeddings_arr, labels)
        return img, label

    @classmethod
    def process_image(cls, img):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [64, 64], method=ResizeMethod.BICUBIC, antialias=True)

    def process_path(self, file_path):
        img = tf.io.read_file(file_path)
        parts = tf.strings.split(file_path, os.path.sep)
        label = bytes.decode(parts[-2].numpy())
        index = self.label_index_dict[label]
        img = self.process_image(img)
        img = img / 255.0
        return img, index
