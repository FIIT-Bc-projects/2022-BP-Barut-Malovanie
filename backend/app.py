import base64
import torch

from flask import Flask, request
from io import BytesIO
from matplotlib import pyplot
from deep_learning.main_model.models import Encoder, Generator

app = Flask(__name__)
device = 'cpu'
model = Generator()
state_dict = torch.load('trained_models/flowers_generator_final', map_location=device)
model.load_state_dict(state_dict)
model.eval()


@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response


@app.route("/", methods=['GET'])
def home():
    prompt = request.args['input']
    print(prompt)
    encoded_prompts = Encoder.encode_prompt([prompt] * 5, device=device)
    encoded_prompts = torch.reshape(encoded_prompts, (*encoded_prompts.shape, 1, 1))
    latent_vec = torch.randn(5, 100, 1, 1, device=device)
    with torch.no_grad():
        images = model(latent_vec, encoded_prompts)
    images = (images + 1) / 2
    response_arr = []

    for image in images:
        tmp = BytesIO()
        pyplot.axis('off')
        pyplot.imshow(image.permute(1, 2, 0))
        pyplot.savefig(tmp, format='png', bbox_inches="tight", transparent=True, pad_inches=0)
        pyplot.close()
        base64_image = base64.encodebytes(tmp.getvalue()).decode('utf-8')
        response_arr.append(base64_image)
    return {"images": response_arr}


if __name__ == "__main__":
    app.run(port=8001)
