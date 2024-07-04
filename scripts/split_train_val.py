"""
split all images and all labels into training and validation
"""

import os
import random
import shutil


images_dir = '../dataset/images'
labels_dir = '../dataset/labels'

tra_img_dir = '../dataset/train/images'
test_img_dir = '../dataset/test/images'
val_img_dir = '../dataset/val/images'

tra_labels_dir = '../dataset/train/labels'
test_labels_dir = '../dataset/test/labels'
val_labels_dir = '../dataset/val/labels'

if not os.path.exists(tra_img_dir):
    os.makedirs(tra_img_dir)
if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)
if not os.path.exists(test_img_dir):
    os.makedirs(test_img_dir)
if not os.path.exists(tra_labels_dir):
    os.makedirs(tra_labels_dir)
if not os.path.exists(val_labels_dir):
    os.makedirs(val_labels_dir)
if not os.path.exists(test_labels_dir):
    os.makedirs(test_labels_dir)



labels_filename = os.listdir(labels_dir)    # ['M1306_000215.txt', 'M0501_000291.txt', ......]
random.shuffle(labels_filename)    # shuffle the dataset

tra_filename = labels_filename[:30000]
test_filename = labels_filename[30000:35000]
val_filename = labels_filename[35000:]

print('number of training images is: ', len(tra_filename))
print('number of validation images is: ', len(val_filename))
print('number of testing images is: ', len(test_filename))

# training dataset
for tra_file in tra_filename:    # 'M1306_000215.txt'
    prefix = tra_file[:12]    # 'M1306_000215'
    old_tra_img = images_dir + '/' + prefix + '.jpg'
    new_tra_img = tra_img_dir + '/' + prefix + '.jpg'
    shutil.copyfile(old_tra_img, new_tra_img)

    old_tra_label = labels_dir + '/' + prefix + '.txt'
    new_tra_label = tra_labels_dir + '/' + prefix + '.txt'
    shutil.copyfile(old_tra_label, new_tra_label)
print('training images have been saved in folder: ', tra_img_dir)
print('training labels have been saved in folder: ', tra_labels_dir)


# testing dataset
for test_file in test_filename:    # 'M1306_000215.txt'
    prefix = test_file[:12]    # 'M1306_000215'
    old_test_img = images_dir + '/' + prefix + '.jpg'
    new_test_img = test_img_dir + '/' + prefix + '.jpg'
    shutil.copyfile(old_test_img, new_test_img)

    old_test_label = labels_dir + '/' + prefix + '.txt'
    new_test_label = test_labels_dir + '/' + prefix + '.txt'
    shutil.copyfile(old_test_label, new_test_label)
print('testing images have been saved in folder: ', test_img_dir)
print('testing labels have been saved in folder: ', test_labels_dir)


# validation dataset
for val_file in val_filename:    # 'M1306_000215.txt'
    prefix = val_file[:12]    # 'M1306_000215'
    old_val_img = images_dir + '/' + prefix + '.jpg'
    new_val_img = val_img_dir + '/' + prefix + '.jpg'
    shutil.copyfile(old_val_img, new_val_img)

    old_val_label = labels_dir + '/' + prefix + '.txt'
    new_val_label = val_labels_dir + '/' + prefix + '.txt'
    shutil.copyfile(old_val_label, new_val_label)
print('validation images have been saved in folder: ', val_img_dir)
print('validation labels have been saved in folder: ', val_labels_dir)

# remove old dataset folders
shutil.rmtree(images_dir)
shutil.rmtree(labels_dir)