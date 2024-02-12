import base64

from tensorflow import keras
from flask import Flask, request
from io import BytesIO
from matplotlib import pyplot
from ex_2_flowers.dataset_initializer import DatasetInitializer
from ex_2_flowers.utils import generate_con_fake_samples

app = Flask(__name__)
model = keras.models.load_model('trained_models/flowers_generator_model_100.h5')
tokenizer = DatasetInitializer.initialize_tokenizer("backend/all_descriptions.txt")

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

@app.route("/", methods=['GET'])
def home():
    prompt = request.args['input']
    print(prompt)
    inputs = DatasetInitializer.process_text_input(tokenizer, [prompt] * 5, 20)
    images, _ = generate_con_fake_samples(model, inputs, 128, len(inputs))
    response_arr = []
    for image in images:
        tmp = BytesIO()
        pyplot.axis('off')
        pyplot.imshow(image[:, :, :])
        pyplot.savefig(tmp, format='png', bbox_inches="tight", transparent=True, pad_inches=0)
        pyplot.close()
        base64_image = base64.encodebytes(tmp.getvalue()).decode('utf-8')
        response_arr.append(base64_image)
    return {"images": response_arr}


if __name__ == "__main__":
    app.run(port=8001)
