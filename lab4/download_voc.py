"""Confirms the PASCAL VOC 2012 segmentation dataset from the Pre-Lab is in place."""
from torchvision.datasets import VOCSegmentation
from torchvision.datasets.utils import download_and_extract_archive

DATA_DIR = './data'
URL = 'https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
MD5 = '6cd6e144f989b92b3379bac3b3de84fd'

# Already downloaded in the Pre-Lab: keep commented out so the dataset is not downloaded again
# download_and_extract_archive(URL, DATA_DIR, md5=MD5)

train = VOCSegmentation(DATA_DIR, year='2012', image_set='train', download=False)
val = VOCSegmentation(DATA_DIR, year='2012', image_set='val', download=False)
print(f'train: {len(train)} val: {len(val)}')
