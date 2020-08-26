import config
import torch
import flask
import time
from flask import Flask
from flask import request
from model import BERTBaseUncased
from collections import OrderedDict
import functools
import torch.nn as nn


app = Flask(__name__)

MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    state_dict = torch.load(config.MODEL_PATH)
    #unexpected key module.* due to model saved as DataParallel model, need to remove module. from each state_dict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        module_name = k.replace('module.','') #remove 'module.'
        new_state_dict[module_name] = v
    MODEL.load_state_dict(new_state_dict)
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host="127.0.0.1", port="9999")
