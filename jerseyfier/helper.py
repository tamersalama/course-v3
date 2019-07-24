from PIL import Image
import os

def remove_bad_files(fnames):
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
                
    return fnames            