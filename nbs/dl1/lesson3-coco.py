from fastai.vision import *
from fastai.datasets import *
import pandas as pd




# planet = untar_data(URLs.PLANET_TINY)
# planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

coco = untar_data(URLs.COCO_TINY)

coco


# json is quiet clear - not sure why get_annotations returns such structures who then needs to be further put into image_name: [[bboxes], [classes]]
images, lbl_bbox = get_annotations(coco/'train.json')

# {'000000299489.jpg': [[[55.86, 86.51, 65.9, 90.10000000000001], [55.79, 90.37, 65.81, 93.18]], ['vase', 'vase']]}
img2bbox = {img:bb for img, bb in zip(images, lbl_bbox)}

get_y_func = lambda o:img2bbox[o.name]

# label_from_func passes a PosixPath to the function
# calling .name on PosixPath returns the name of the file


data = (ObjectItemList.from_folder(coco).random_split_by_pct().label_from_func(get_y_func).transform(get_transforms(), tfm_y=True).databunch(bs=16, collate_fn=bb_pad_collate))

data.show_batch(rows=5, ds_type=DatasetType.Valid, figsize=(10,10))

# show sample image
# show_image(image_list.open(coco/'train/000000070459.jpg'))



img2bbox[coco/'train/000000070459.jpg']






# data = (ImageFileList.from_folder(coco)
#         #Where are the images? -> in coco
#         .label_from_func(get_y_func)                    
#         #How to find the labels? -> use get_y_func
#         .random_split_by_pct()                          
#         #How to split in train/valid? -> randomly with the default 20% in valid
#         .datasets(ObjectDetectDataset)                  
#         #How to create datasets? -> with ObjectDetectDataset
#         #Data augmentation? -> Standard transforms with tfm_y=True
#         .databunch(bs=16, collate_fn=bb_pad_collate))   
#         #Finally we convert to a DataBunch and we use bb_pad_collate


# path = Path('/floyd/input/planet_amazon')

# df = pd.from_csc(path / 'train_v2.csv')
# df.head()



# i = ImageList.from_folder(path)

# train_files = (path / 'train-jpg').ls()
# train_img_p = list(map(lambda x: str(x), train_files))

# import re
# p = re.compile(r'jpg/', re.IGNORECASE)
# p.match("hello.jpg")[0]


