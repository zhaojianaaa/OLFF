from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
import copy

def get_max_patch(mask):
    mask[mask > 0] = 1
    #labeled_img, num = label(mask, neighbors=4, background=0, return_num=True)
    labeled_img, num = label(mask, connectivity=1, background=0, return_num=True)
    vals = np.unique(labeled_img)
    if 0 in vals and vals[0] == 0:
        vals = vals[1:]
    nums = [len(np.where(labeled_img == val)[0]) for val in vals]
    
    labeled_img[labeled_img != vals[np.argmax(nums)]] = 0
    labeled_img[labeled_img == vals[np.argmax(nums)]] = 1
    yy, xx = np.where(labeled_img > 0) 
    #return x_min,y_min,x_max,y_max
    return np.min(xx), np.min(yy), np.max(xx), np.max(yy), labeled_img


def get_max_patch1(mask):
    mask[mask > 0] = 1
    labeled_img, num = label(mask, neighbors=4, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(0, num):
        if np.sum(labeled_img == 1) > max_num:
            max_num = np.sum(labeled_img == 1)
            max_label = i
    #mcr = (labeled_img == max_label)

    max_area = 0
    for region in regionprops(labeled_img):
        # skip small images
        if region.area < 10:
            continue
        #print(regionprops(labeled_img)[max_label])
        minr, minc, maxr, maxc = region.bbox
        if region.area >= max_area:
            max_area = region.area
            y_min = minr
            x_min = minc
            y_max = maxr
            x_max = maxc

    return x_min,y_min,x_max,y_max

'''
img_paths = os.listdir('./images/')

for i in range(len(img_paths)):
    pathi = './images/' + img_paths[i]
    img_pathis = os.listdir(pathi)
    for j in range(len(img_pathis)):
        pathij = './images/' + img_paths[i] + '/' + img_pathis[j]
        if pathij[-14:] == '_bbox_mask.png':
            pngj = io.imread(pathij)
            [x0,y0,x1,y1] = get_max_patch(pngj)
            x0 = 0 if x0 < 0 else x0
            x1 = 0 if x1 < 0 else x1
            y0 = 0 if y0 < 0 else y0
            y1 = 0 if y1 < 0 else y1
            x0 = 36 if x0 > 36 else x0
            x1 = 36 if x1 > 36 else x1
            y0 = 36 if y0 > 36 else y0
            y1 = 36 if y1 > 36 else y1
            pngji = copy.deepcopy(pngj)
            pngji[y0, x0:x1] = 10
            pngji[y1, x0:x1] = 10
            pngji[y0:y1, x0] = 10
            pngji[y0:y1, x1] = 10
            plt.subplot(1,2,1)
            plt.imshow(pngj)
            plt.subplot(1,2,2)
            plt.imshow(pngji)
            plt.show()
            a = 0
'''

        


