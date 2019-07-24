from fastai.vision import *
from fastai.metrics import error_rate
np.random.seed(2)
batch_size = 64


# Training

# Prepare Data
# Untar data to path
path = untar_data(URLs.PETS)
# path.ls()
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)

# Pattern to extract labels through match[1]
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=batch_size)
# data.show_batch(rows=3, figsize=(7,6))
data.normalize(vision.imagenet_stats)



learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(3)



# Saving Model
learn.model_dir = Path('/floyd/home')
learn.save('stage-1')



# Loading model
# Doesn't look it's possible without accss to data
# ??
# import torch
# torch.load(PATH)
learn.load('stage-1')

# Viewing classified data
interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()
# len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused()


# Retrain with default learning rate
learn.unfreeze()
learn.fit_one_cycle(3)

# Retrain with discovered learning rate (better)
learn.load('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
# Try viewing & most confused


# Resnet-50
learn_50 = create_cnn(data, models.resnet50, metrics=error_rate)
learn_50.model_dir = Path('/floyd/home')

# Fit
learn_50.fit_one_cycle(3)

# Save
learn_50.save('stage-1-50')

# Load saved model
learn_50.load('stage-1-50')


# Viewing classified data
interp = ClassificationInterpretation.from_learner(learn_50)

# losses,idxs = interp.top_losses()
# len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused()

# Find learning rate
learn_50.unfreeze()
learn_50.lr_find()
learn_50.recorder.plot()
learn_50.fit_one_cycle(4, max_lr=slice(1e-6, 1e-3))

# Saving the new model
learn_50.save('stage-2-50')
