import os
from PIL import Image
from PIL import ImageFile
import cv2
import os


def reduce(place):
    if (os.stat(place).st_size*0.001) > 100:
        image = Image.open(place)
        try:
            if ((os.stat(place).st_size*0.001) > 100) and ((os.stat(place).st_size*0.001) < 300):
                image.save(place, quality=90, optimize=True)
            else:
                image.save(place, quality=50, optimize=True)

        except:
            image = image.convert("RGB")
            # image.save(place, quality=100, optimize=True)
            if ((os.stat(place).st_size*0.001) > 100) and ((os.stat(place).st_size*0.001) < 300):
                image.save(place, quality=90, optimize=True)
            else:
                image.save(place, quality=50, optimize=True)

        print(place)


def resize1(place):
    img = cv2.imread(place)
    size = img.shape[0]*img.shape[1]*img.shape[2]*0.001
    if size > 50:
        scale_percent = 40  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(place, img)


def get_all_files(path):
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            # print(f)
            # fname.append(os.path.join(root, f))
            if ('.png' in f.lower()) or ('.jpg' in f.lower()) or ('.jpeg' in f.lower()):
                # print(f)

                mainpath = os.path.join(root, f)
                # print(mainpath)
                reduce(mainpath)


get_all_files('/home/system/greycampus_v19/app/assets/images/')
