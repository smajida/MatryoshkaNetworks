#!/usr/bin/env python2.7

from __future__ import print_function
import argparse
import cv2
import numpy
import os
from os.path import exists, join
from time import time

def convert_to_npy(img_path, out_dir, block_size=50000):
    print("Converting images in {} to npy files -- center cropped and resized to 64x64:".format(img_path))
    f_list = os.listdir(img_path)
    img_files = ["{}/{}".format(img_path,f_name) for f_name in f_list \
                 if ((f_name.find('jpg') > -1) or (f_name.find('jpeg') > -1))]
    start_time = time()
    im_count = 0
    total_ims = 0
    npy_file_count = 0
    im_ary = numpy.zeros((block_size, 3, 64, 64), dtype=numpy.uint8)
    for i, f_name in enumerate(img_files):
        # load image from file
        img = cv2.imread(f_name, cv2.IMREAD_COLOR)
        if (i == 0):
            print("-- image file: {}".format(f_name))
            print("-- uncropped shape: {}".format(img.shape))
        # center crop the image to be square shaped
        max_axis = 0
        min_axis = 1
        if img.shape[1] > img.shape[0]:
            max_axis = 1
            min_axis = 0
        max_dim = img.shape[max_axis]
        min_dim = img.shape[min_axis]
        crop_shift = (max_dim - min_dim) // 2
        if max_axis == 0:
            img = img[crop_shift:(crop_shift+min_dim),:,:]
        else:
            img = img[:,crop_shift:(crop_shift+min_dim),:]
        # resize the image to 64x64
        img2 = cv2.resize(img, dsize=(64,64))
        img2 = numpy.swapaxes(img2, 0, 2)
        img2 = numpy.swapaxes(img2, 1, 2)
        # little safety check on image shape
        if (i == 0):
            print("-- cropped shape: {}".format(img.shape))
            print("-- resized shape: {}".format(img2.shape))
        # swap color axes, to get RGB instead of BGR (like, WTF?)
        temp_img = numpy.copy(img2)
        img2[0,:,:] = temp_img[2,:,:]
        img2[2,:,:] = temp_img[0,:,:]
        im_ary[im_count, :, :, :] = img2
        im_count += 1
        total_ims += 1
        if (total_ims % 1000) == 0:
            print("    processed {} images...".format(total_ims))
        if im_count == block_size:
            # write a block of center cropped, resized images to disk.
            npy_file_count += 1
            file_name = "{}/imgs_64x64_{}.npy".format(out_dir, npy_file_count)
            numpy.save(file_name, im_ary)
            print("    -- wrote {} images to {}.".format(im_count, file_name))
            im_count = 0
            im_ary = numpy.zeros((block_size, 3, 64, 64), dtype=numpy.uint8)
    # handle final block of images (this block is generally smaller...)
    npy_file_count += 1
    file_name = "{}/imgs_64x64_{}.npy".format(out_dir, npy_file_count)
    numpy.save(file_name, im_ary[:im_count,:,:,:])
    print("    -- wrote {} images to {}.".format(im_count, file_name))
    end_time = time()
    print("processed {} images in {} seconds.".format(total_ims, (end_time - start_time)))
    return


if __name__ == '__main__':
    convert_to_npy('/NOBACKUP/faces_celeba/img_align_celeba',
                   '/NOBACKUP/faces_celeba/imgs_as_npy',
                   block_size=50000)
