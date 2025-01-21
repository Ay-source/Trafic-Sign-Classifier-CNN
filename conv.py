import os
import glob

from PIL import Image


directory = "./data/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/*.ppm"
new_dir = "./data/test/"
def conv_ppm_to_jpg(directory, new_dir):
    ppm_files = glob.glob(directory)
    for ppm_file in ppm_files:
        img = Image.open(ppm_file)
        new_dir = new_dir + os.path.basename(ppm_file) \
        .strip(".ppm") + ".jpg"
        with open(new_dir,  'w'):
            pass
        img.save(new_dir, "JPEG")


conv_ppm_to_jpg(directory, new_dir)