import torch
torch.cuda.device("cpu")

# add your custom labels
labels = ['boston_bruins', 'calgary_flames', 'chicago_blackhawks', 'detroit_red_wings', 'edmonton_oilers', 'montreal_canadiens', 'new_york_rangers', 'ottawa_senators', 'philadelphia_flyers', 'pittsburg_penguins', 'toronto_maple_leafs', 'vancouver_canucks']
	

# set your data directory
data_dir = 'data'

# set the URL where you can download your model weights
MODEL_URL = '/floyd/home/hockey-unfrozen-unnormalized-w_lr'

# set some deployment settings
# PORT = 8080


from flask import Flask, jsonify, request
import logging
import random
import time

from PIL import Image
import requests, os
from io import BytesIO

# import fastai stuff
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import fastai
defaults.device = torch.device('cpu')
np.random.seed(2)
#batch_size = 64
batch_size = 32



# import settings
# from settings import *


# set dir structure
#def make_dirs(labels, data_dir):
#    root_dir = os.getcwd()
#    make_dirs = ['train', 'valid', 'test']
#    for n in make_dirs:
#        name = os.path.join(root_dir, data_dir, n)
#        for each in labels:
#            os.makedirs(os.path.join(name, each), exist_ok=True)
#make_dirs(labels=labels, data_dir=data_dir) # comes from settings.py

path = Path(data_dir)


# set flask params

app = Flask(__name__)


def load_model():
    # download model weights if not already saved
    path_to_model = Path(MODEL_URL)
    print('done!\nloading up the saved model weights...')
    #fastai.defaults.device = torch.device('cpu') # run inference on cpu
    #empty_data = ImageDataBunch.single_from_classes(path, labels, tfms=get_transforms(), size=224).normalize(imagenet_stats)
    #learn = create_cnn(empty_data, models.resnet34)
    # model = torch.load(MODEL_URL + '.pth', map_location='cpu')
    #model.eval()
    defaults.device = torch.device('cpu')
    print('defaults device')
    print(defaults.device)
    learner = load_learner('/floyd/home/', fname='model.pkl')
    return learner
    
#    learn = learn.load(MODEL_URL)
#    return learn
    
    
    
model = load_model()   

@app.route("/")
def hello():
    return "Image classification example\n"

@app.route('/predict', methods=['GET'])
def predict():
    url = request.args['url']
    app.logger.info("Classifying image %s" % (url),)
    
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    
    trn_tfms, val_tfms = get_transforms(models.resnet50, batch_size)
    img = img.apply_tfms(val_tfms)

    
    t = time.time() # get execution time
    pred_class, pred_idx, outputs = model.predict(img)
    dt = time.time() - t
    app.logger.info("Execution time: %0.02f seconds" % (dt))
    app.logger.info("Image %s classified as %s" % (url, pred_class))
    #return jsonify(pred_class)
    return jsonify(str(pred_class))

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=False)