from fastai.vision import *
import pandas as pd


# planet = untar_data(URLs.PLANET_TINY)
# planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


path = Path('/floyd/input/planet_amazon')

df = pd.from_csc(path / 'train_v2.csv')
df.head()



i = ImageList.from_folder(path)

train_files = (path / 'train-jpg').ls()
train_img_p = list(map(lambda x: str(x), train_files))

import re
p = re.compile(r'jpg/', re.IGNORECASE)
p.match("hello.jpg")[0]


