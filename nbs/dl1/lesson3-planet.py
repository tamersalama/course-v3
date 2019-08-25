from fastai.vision import *
import pandas as pd
np.random.seed(42)


planet_path = untar_data(URLs.PLANET_TINY)
# planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
# path = Path('/floyd/input/planet_amazon')

# df = pd.read_csv(planet_path / 'labels.csv')
# df.head()



# i = ImageList.from_folder(path)

# data = ImageList.from_csv(planet_path, 'labels.csv', folder='train', suffix='.jpg').\
#         split_by_rand_pct().\
#         label_from_df(label_delim=' ').\
#         databunch()



data = ImageList.from_csv(planet_path, 'labels.csv', folder='train', suffix='.jpg').split_by_rand_pct(0.2).label_from_df(label_delim = ' ')

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


data = (data.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))


data.show_batch(rows=4, figsize=(10, 7))

arch = models.resnet50

acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])


learn.lr_find()


learn.recorder.plot()

lr = 1e-01

learn.fit_one_cycle(10, slice(lr))

#
# train_files = (planet_path / 'train-jpg').ls()
# train_img_p = list(map(lambda x: str(x), train_files))
#
# import re
# p = re.compile(r'jpg/', re.IGNORECASE)
# p.match("hello.jpg")[0]


