#!/usr/bin/env python2.7

from __future__ import print_function
import argparse
import cv2
import lmdb
import numpy
import os
from os.path import exists, join
from time import time

__author__ = 'Fisher Yu'
__email__ = 'fy@cs.princeton.edu'
__license__ = 'MIT'


def view(db_path):
    print('Viewing', db_path)
    print('Press ESC to exist or SPACE to advance.')
    window_name = 'LSUN'
    cv2.namedWindow(window_name)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            print('Current key:', key)
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8),
                cv2.IMREAD_COLOR)
            cv2.imshow(window_name, img)
            c = cv2.waitKey()
            if c == 27:
                break

def convert_to_npy(db_path, out_dir, block_size=100000, center_crop=True):
    print("Converting images in {} to npy files:".format(db_path))
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    start_time = time()
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        im_count = 0
        npy_file_count = 0
        im_ary = numpy.zeros((block_size, 3, 64, 64), dtype=numpy.uint8)
        for key, val in cursor:
            img = cv2.imdecode(
                numpy.fromstring(val, dtype=numpy.uint8),
                cv2.IMREAD_COLOR)
            if center_crop and ((img.shape[0] > 256) or (img.shape[1] > 256)):
                # imgs are presized to always have min axis dim of 256
                max_axis = 0
                if img.shape[1] > img.shape[0]:
                    max_axis = 1
                max_dim = img.shape[max_axis]
                crop_shift = (max_dim - 256) // 2
                if max_axis == 0:
                    img = img[crop_shift:(crop_shift+256),:,:]
                else:
                    img = img[:,crop_shift:(crop_shift+256),:]
            img2 = cv2.resize(img, dsize=(64,64))
            img2 = numpy.swapaxes(img2, 0, 2)
            img2 = numpy.swapaxes(img2, 1, 2)
            # swap color axes, to get RGB instead of BGR (like, WTF?)
            temp_img = numpy.copy(img2)
            img2[0,:,:] = temp_img[2,:,:]
            img2[2,:,:] = temp_img[0,:,:]
            im_ary[im_count, :, :, :] = img2
            im_count += 1
            if (im_count % 1000) == 0:
                print("    processed {} images...".format(im_count))
            if im_count == block_size:
                # write a block of resized images to disk.
                npy_file_count += 1
                file_name = "{}/imgs_64x64_{}.npy".format(out_dir, npy_file_count)
                numpy.save(file_name, im_ary)
                print("    -- wrote {} images to {}.".format(im_count, file_name))
                im_count = 0
                im_ary = numpy.zeros((block_size, 3, 64, 64), dtype=numpy.uint8)
    # handle final block of images in lmdb file (this block is smaller...)
    npy_file_count += 1
    file_name = "{}/imgs_64x64_{}.npy".format(out_dir, npy_file_count)
    numpy.save(file_name, im_ary[:im_count,:,:,:])
    print("    -- wrote {} images to {}.".format(im_count, file_name))
    im_ary = numpy.zeros((block_size, 3, 64, 64), dtype=numpy.uint8)
    end_time = time()
    print("processed {} images in {} seconds.".format(im_count, (end_time - start_time)))
    return

def export_images(db_path, out_dir):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776,
                    max_readers=100, readonly=True)
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            image_out_dir = join(out_dir, '/'.join(key[:6]))
            if not exists(image_out_dir):
                os.makedirs(image_out_dir)
            image_out_path = join(image_out_dir, key + '.jpg')
            with open(image_out_path, 'w') as fp:
                fp.write(val)
            count += 1
            if count % 1000 == 0:
                print('Finished', count, 'images')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', type=str,
                        choices=['view', 'convert', 'export'],
                        help='view: view the images in the lmdb database '
                             'interactively.\n'
                             'export: Export the images in the lmdb databases '
                             'to a folder. The images are grouped in subfolders'
                             ' determinted by the prefiex of image key.')
    parser.add_argument('lmdb_path', nargs='+', type=str,
                        help='The path to the lmdb database folder. '
                             'Support multiple database paths.')
    parser.add_argument('--out_dir', type=str, default='')
    args = parser.parse_args()

    command = args.command
    lmdb_paths = args.lmdb_path

    for lmdb_path in lmdb_paths:
        if command == 'view':
            view(lmdb_path)
        elif command == 'export':
            export_images(lmdb_path, args.out_dir)
        elif command == 'convert':
            convert_to_npy(lmdb_path, args.out_dir)


if __name__ == '__main__':
    main()
