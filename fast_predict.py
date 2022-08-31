
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import os, glob

target_size = [256, 256] # size of image for model
shuffle_buffer = 32 
batch_size = 4 # batch size
MODEL_DIR = './' #dir where saving models or weights
test_batch = 32 #images in batch to predict

train_dir = 'train_v2' # folder with train image
train_mask_dir = 'train_masks' # folder with masks
test_dir = 'test_v2' # folder with test image

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'airbus_model.h5'))

def get_predict_mask_list(imgs, threshold = 0.5, pred_size = 256):
    """
    Predict mask
    imgs - list of imags 
    threshold - threshold for classes
    pred_size - size of window for cut

    Return list of masks (array)
    """
    img_shape = (imgs[0].shape[0],imgs[0].shape[1])
    # how many parts cut the picture, it will be necessary to create a mask
    mask_parts = [np.ceil(img_shape[0]/pred_size).astype(int), np.ceil(img_shape[1]/pred_size).astype(int)]
    all_parts = mask_parts[0]*mask_parts[1]

    stack_parts = []
    # Pass through the analyzed image, pulling out sections of pred_size in size
    for img in imgs:
        for i in range(0, img_shape[0], pred_size):
            for j in range(0, img_shape[1], pred_size):
                patch_img = img[i:i+pred_size, j:j+pred_size]
                patch = np.zeros((pred_size, pred_size, 3))
                patch[:patch_img.shape[0], :patch_img.shape[1]] = patch_img

                stack_parts.append(patch / 127.5 - 1)

    # predict mask
    pred = model.predict(np.array(stack_parts))
    #Use threshold
    pred[pred>=threshold] = 1
    pred[pred<threshold] = 0

    full_masks = []
    for g in range(len(imgs)):

        #mask with a multiple of pred_size sides
        full_mask = np.zeros((mask_parts[0]*pred_size, mask_parts[1]*pred_size))

        for k in range(all_parts):
            i = k//mask_parts[0]*pred_size
            j = k%mask_parts[1]*pred_size
            full_mask[i:i+pred_size, j:j+pred_size] = np.squeeze(pred[g*all_parts+k])

        # Crop the mask to fit the image
        full_mask = full_mask[:img.shape[0], :img.shape[1]]
        full_masks.append(full_mask)
    return full_masks


def decode_mask(mask, shape=(768, 768)):
    #from image(mask) to run-length encoding
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#predict all test images
test_batch = 32
test_data = os.listdir(test_dir)
ship_list_dict = []
for i in tqdm(range(len(test_data)//test_batch+1)):
    batch_imgs = test_data[i*test_batch:i*test_batch+test_batch]
    imgs = []
    for img_name in batch_imgs:
        imgs.append(plt.imread(os.path.join(test_dir, img_name)))
    
    masks = get_predict_mask_list(imgs,0.8)
    for img_name, predict_mask in zip(batch_imgs, masks):
        decode = decode_mask(predict_mask)
        ship_list_dict.append({'ImageId':img_name,'EncodedPixels':decode})

pred_df = pd.DataFrame(ship_list_dict)
pred_df.to_csv('submission.csv', index=False)
pred_df

#load submission to kaggle
#!kaggle competitions submit -c airbus-ship-detection -f submission.csv -m "20 epoch, treshold=0.5"