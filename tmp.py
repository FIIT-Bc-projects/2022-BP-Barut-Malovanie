from matplotlib import pyplot
from tensorflow import keras
import numpy as np

model = keras.models.load_model("saved/generator_model_001.h5")
x_input = np.random.randn(100)
x_input = x_input.reshape(1, 100)
X = model.predict(x_input)
pyplot.axis('off')
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.savefig('./generated_image.png', bbox_inches="tight", transparent=True, pad_inches=0)
pyplot.close()