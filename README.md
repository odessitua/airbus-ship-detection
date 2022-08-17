# airbus-ship-detection
https://www.kaggle.com/competitions/airbus-ship-detection/overview

Task was solved in colab, easy to test here: 
https://colab.research.google.com/drive/1kKh8gFHYckXl4tHxKqgwoAgk6Y5-mPv4?usp=sharing

# Features

### 1. Used DeepLabV3+ model with trained ResNet50 

https://keras.io/examples/vision/deeplabv3_plus/

Masks generated as png files


### 2. Class balance

Segmentation is presented as a binary classification (there is a ship or not), and we have a unbalance of classes even in one picture where there is a ship. Since class_weight cannot be used with output folds, I wrote own loss (weighted_binary_crossentropy), where weight can be set for one class.
Also I deleted pictures where there are no ships.

### 3. Augmentation

When augmenting, I first accidentally cut out a piece of 256 * 256. After that, I randomly flip horizontally, vertically and rotate 90 degrees (8 positions). Do the same for the mask.

I also change a little brightness, contrast, saturation, add noise.
There is no resizing, so there is no loss of information, plus it is possible to learn from pictures of any size. Also, because of the crop, a lot of pictures without ships are formed, so the original pictures without them are not needed.

### 4. Gradient Centralization
https://arxiv.org/abs/2004.01461

### 5. Checkpoin in training, threshold at predict. Predict generated by 9 parts of image

# Files
- data_loader.py - Load data from kaggle, need kaggle.json 
- create_masks.py - (optional) Generate png masks from csv, need train_v2, train_ship_segmentations_v2.csv
- train.py - Create and train model, output airbus_model.h5
- predict.py - Predict masks, output submission.csv
- airbus_model.h5 - trained model
- submission.csv - civ file for gaggle competition (private 0.77330, public 0.58010)
- Kaggle_airbus_ship_detection.ipynb - copy of colab (Jupyter) notebook: https://colab.research.google.com/drive/1kKh8gFHYckXl4tHxKqgwoAgk6Y5-mPv4?usp=sharing
- train_masks.zip - archive png-files with masks
- requirements.txt (used pip freeze > requirements.txt in colab)

# Results 
I get a good score in first try.

### What to do for best score
- try to change brightness-contrast augmentation
- different weights in loss weighted_binary_crossentropy
- different threshold in predict
- try different learning rate with reduce
- try different count of epoch without checkpoints (save weights in each epoch and check them)


