from flask import Flask, render_template
from io import BytesIO
import base64
from matplotlib import pyplot
import matplotlib
matplotlib.use('Agg')
from tensorflow import keras
import numpy as np
app = Flask(__name__)
model = keras.models.load_model("../trained_models/trained_model.h5")


@app.route("/", methods=['GET', 'POST'])
def home():
    image = BytesIO()
    x_input = np.random.randn(100)
    x_input = x_input.reshape(1, 100)
    X = model.predict(x_input)
    pyplot.axis('off')
    pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
    pyplot.savefig(image, format='png', bbox_inches="tight", transparent=True, pad_inches=0)
    pyplot.close()
    base_64_image = base64.encodestring(image.getvalue())
    return render_template('template.html', image = base_64_image.decode('utf-8'))


if __name__ == "__main__":
    app.run()
