from fastai.vision import *
from fastai.metrics import error_rate
np.random.seed(2)
#batch_size = 64
batch_size = 32


# Training

# Prepare Data
# Untar data to path
# path = untar_data(URLs.PETS)
# path_anno = path/'annotations'
# path_img = path/'images'
# fnames = get_image_files(path_img)

data_dir = '/floyd/input/hockey_jerseys'
fnames = get_image_files(data_dir, recurse=True); fnames

verify_images(data_dir, delete=False, )


len(fnames)
from PIL import Image
import os
for f in fnames[:]:
    img = None
    try:
        img = Image.open(f)
        img.verify()
        img.close()
    except:
        if img != None:
            img.close()
        fnames.remove(f)
        # print(f"bad file {f}")
len(fnames)


# Pattern to extract labels through match[1]
# pat = r'/([^/]+)_\d+.jpg$'
#data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=batch_size)
# pat = r'\/hockey_jerseys\/(.*)\/'
# data = ImageDataBunch.from_name_re(data_dir, fnames, pat, ds_tfms=get_transforms(), size=440, bs=batch_size)

pat = r'\/hockey_jerseys\/(.*)\/'
# load data benchmark
import time
start = time.time()
data = ImageDataBunch.from_name_re(data_dir, fnames, pat, ds_tfms=get_transforms(), size=440, bs=batch_size)
end = time.time()
len(fnames)
# data.show_batch(rows=3, figsize=(7,6))
read_time = end - start
data.normalize(vision.imagenet_stats)


learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir='/floyd/home')
#learn.model_dir = Path('/floyd/home')
start = time.time()
learn.fit_one_cycle(6)
end = time.time()
training_time = end-start


# Saving Model
learn.save(f"hockey-stage-{}")



# Loading model
# Doesn't look it's possible without accss to data
# ??
# import torch
# torch.load(PATH)
learn.load('hockey-stage-1')

# Viewing classified data
interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()
# len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused()


# Retrain with default learning rate
learn.unfreeze()

start = time.time()
learn.fit_one_cycle(6)
end = time.time()
training_time_after_unfreeze = end-start


# Retrain with discovered learning rate (better)
learn.load('hockey-stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6, max_lr=slice(1e-3,1e-2))
# Try viewing & most confused

learn.save('hockey-unnormalized_unfrozen_w_lr')

model = torch.load('/floyd/home/hockey-unnormalized_unfrozen_w_lr.pth')


# Resnet-50
learn_50 = create_cnn(data, models.resnet50, metrics=error_rate, model_dir=Path('/floyd/home'))
#learn_50.model_dir = Path('/floyd/home')

# Fit

learn_50.lr_find()
learn_50.recorder.plot()
learn_50.fit_one_cycle(6, max_lr=slice(1e-3, 0.05))


learn_50.unfreeze()
learn_50.lr_find()
learn_50.fit_one_cycle(6, max_lr=slice(1e-3, 1e-2))
 
import dill as dill
new_model_path = Path('/floyd/home/hockey-unfrozen-unnormalized-w_lr.pth')
torch.save(learn_50.model, new_model_path, pickle_module=dill)


model = torch.load(new_model_path)
model.eval()



# Save
learn_50.save('stage-1-50')

# Load saved model
learn_50.load('stage-1-50')


# Viewing classified data
interp = ClassificationInterpretation.from_learner(learn_50)

# losses,idxs = interp.top_losses()
# len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=120)
interp.most_confused()

# Find learning rate
learn_50.unfreeze()
learn_50.lr_find()
learn_50.recorder.plot()
learn_50.fit_one_cycle(4, max_lr=slice(1e-6, 1e-3))

# Saving the new model
learn_50.save('stage-2-50')






learn_50.export('/floyd/home/model.pkl')


ts = load_learner('/floyd/home', fname='model.pkl')



# Prediction

#from fastai.transforms import tfms_from_model
trn_tfms, val_tfms = get_transforms(models.resnet50, batch_size)
img = open_image('/floyd/home/flames1.jpg').apply_tfms(val_tfms)

#model = torch.load(new_model_path)
#model.eval()

ts.predict(img)

log_pred = model.predict_array(img[None])
pred = np.argmax(log_pred, axis=1)
category = data.classes[pred]