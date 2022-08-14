import numpy as np 
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os,glob

train_dir = 'train_v2' # folder with train image
train_mask_dir = 'train_masks' # folder with masks (to save masks)

masks_df = pd.read_csv('train_ship_segmentations_v2.csv')

def save_mask(img_path, shape=(768, 768)):
    '''
    Save png mask
    img_path: path to image
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    image_name = os.path.basename(img_path)
    img_masks = masks_df.loc[masks_df['ImageId'] == image_name, 'EncodedPixels'].tolist()

    all_masks = np.zeros(shape)
    for mask in img_masks:
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        if (pd.isna(mask) != True):
            s = mask.split()
            starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
            starts -= 1
            ends = starts + lengths
            
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1
        all_masks += img.reshape(shape).T  
    #convert to 1 bit png
    im = Image.fromarray(all_masks*256).convert("1")
    im.save(f'{train_mask_dir}/{image_name}.png','PNG')

!mkdir '{train_mask_dir}'

train_images = sorted(glob.glob(os.path.join(train_dir, '*.jpg')))
for image_name in tqdm(train_images):
    save_mask(image_name)

!zip -r train_masks.zip '{train_mask_dir}'

!cp train_masks.zip '/content/drive/MyDrive/Colab Notebooks/Базы/train_masks.zip'