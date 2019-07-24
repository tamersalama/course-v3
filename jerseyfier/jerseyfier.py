from fastai.vision import *
from fastai.metrics import error_rate
from helper import *
import time

t1 = time.time()

# Some variables
np.random.seed(2)
batch_size = 24
data_dir = '/floyd/input/hockey_jerseys'


# Training

# Image file names
fnames = get_image_files(data_dir, recurse=True);


# remove non-images from file names
fnames = remove_bad_files(fnames)


# class identification from file name pattern 
pat = r'\/hockey_jerseys\/(.*)\/'

# Create a data collection
data = ImageDataBunch.from_name_re(data_dir, fnames, pat, ds_tfms=get_transforms(), size=440, bs=batch_size)

# normalize the data
data.normalize(vision.imagenet_stats)


# Create a CNN Network (resnet-50)
learn_50 = create_cnn(data, models.resnet50, metrics=error_rate, model_dir=Path('/floyd/home'))


# Fit (first round of transfer learning learning from images)
learn_50.fit_one_cycle(6)

t2 = time.time()
print("\n\n\n------------")
print(f"Frozen: {t2-t1}")


t1 = time.time()

#  Remove lock on previous layers
learn_50.unfreeze()

# Learn some more
learn_50.fit_one_cycle(6, max_lr=slice(1e-3, 1e-2))

# Viewing classified data and classification errors
# interp = ClassificationInterpretation.from_learner(learn_50)
# interp.plot_top_losses(9, figsize=(15,11))
# interp.plot_confusion_matrix(figsize=(12,12), dpi=120)

# Save the generated model
learn_50.export(f"/floyd/home/jerseyfier/models/model-{int(time.time())}.pkl")

t2 = time.time()
print("\n\n\n------------")
print(f"Frozen: {t2-t1}")
