
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

train_dir = 'train_v2' # folder with train image
train_mask_dir = 'train_masks' # folder with masks
test_dir = 'test_v2' # folder with test image

!unzip train_masks.zip -d '{train_mask_dir}'

masks_df = pd.read_csv('train_ship_segmentations_v2.csv')

def read_image(image_path, mask=False):
    '''
    Function to upload an image.
    image_path - image path
    mask - flag determines whether a image or mask should be output
    output tf.image
    '''
    img_file = tf.io.read_file(image_path)
    if mask == False:
        image = tf.image.decode_jpeg(img_file, channels=3)
        image = tf.cast(image, tf.float32) / 127.5 - 1
    if mask:
        image = tf.image.decode_png(img_file, channels=1)
        # image = tf.squeeze(image)
        image = tf.cast(image, tf.float32) / 256 
    
    return image


def load_data(image_list, mask_list):
    # Loading image and masks
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    image, mask = random_jitter(image, mask)    #augmentation
    return image, mask

def get_dataset(image_list, mask_list):
    # Loading dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    return dataset

@tf.function()
def random_jitter(input_image, output_image):
    #augmentation

    # random crop of image
    input_image, output_image = random_crop(input_image, output_image, target_size[0], target_size[1])

    if tf.random.uniform(()) > 0.5:
    # random horisontal flip
        input_image = tf.image.flip_left_right(input_image)
        output_image = tf.image.flip_left_right(output_image)

    if tf.random.uniform(()) > 0.5:
    # random vertical flip
        input_image = tf.image.flip_up_down(input_image)
        output_image = tf.image.flip_up_down(output_image)

    if tf.random.uniform(()) > 0.5:
    # random rotation 90 degrees
        input_image = tf.image.rot90(input_image)
        output_image = tf.image.rot90(output_image)

    #brightness 10%, contrast 10%, saturation 20% 
    input_image = tf.image.random_brightness(input_image, 0.1)
    input_image = tf.image.random_contrast(input_image, 0.9, 1.1)
    input_image = tf.image.random_saturation(input_image, 0.8,1.2)
    #small hue shift
    input_image = tf.image.random_hue(input_image, 0.01)
    input_image = tf.clip_by_value(input_image,-1,1) # set values from -1 to 1
    #output_image = tf.clip_by_value(output_image,0,1)

    if tf.random.uniform(()) > 0.1:
    # random noise
        noise = tf.random.uniform((input_image.shape), .7, 0.99)
        input_image = tf.cast(input_image, tf.float32) * noise

    return input_image, output_image

def random_crop(input_image, output_image, height, width):
    '''
    Random crop image
    input_image - image
    output_image - masks
    height, width - size to crop
    '''
    # Image stack for synchronous cropping
    stacked_image = tf.concat([input_image, output_image], axis=-1)

    # random crop
    cropped_image = tf.image.random_crop(
        stacked_image, size=[height, width, 4])
    
    # unstack image and mask
    return cropped_image[..., :-1], tf.expand_dims(cropped_image[..., -1], -1)

#size for validation data
val_split = len(os.listdir(train_dir)) // 10

#Loading train dataset
train_images = sorted(glob.glob(os.path.join(train_dir, '*.jpg')))[:-val_split]
train_masks = sorted(glob.glob(os.path.join(train_mask_dir, '*.png')))[:-val_split]

#Loading validation dataset
val_images = sorted(glob.glob(os.path.join(train_dir, '*.jpg')))[-val_split:]
val_masks = sorted(glob.glob(os.path.join(train_mask_dir, '*.png')))[-val_split:]

def cut_empty(images,masks):
    #delete images and mask, where is not ships
    images_lst = []
    masks_lst = []
    df_tmp = masks_df.dropna().set_index('ImageId')
    for i in tqdm(range(len(images))):
        path_i = images[i]
        path_m = masks[i]
        name = os.path.basename(path_i)
        if(name in df_tmp.index):
            images_lst.append(path_i)
            masks_lst.append(path_m)
    return images_lst, masks_lst

train_images,train_masks = cut_empty(train_images,train_masks)
val_images,val_masks = cut_empty(val_images,val_masks)

# create train and validation dataset
train_dataset = get_dataset(train_images, train_masks)
val_dataset = get_dataset(val_images, val_masks)


print("Train Dataset lenght:", len(train_dataset))
print("Val Dataset lenght:", len(val_dataset))
print("\nTrain Dataset:", (train_dataset))
print("Val Dataset:", (val_dataset))

#Model DeepLabV3+
#https://keras.io/examples/vision/deeplabv3_plus/

def convolution_block(
    x,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding='same',
    use_bias=True,
    kernel_regularizer=None
):
    x = layers.Conv2D(filters=num_filters,
                     kernel_size=kernel_size,
                     dilation_rate=dilation_rate,
                     padding=padding,
                     use_bias=use_bias,
                     kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.BatchNormalization()(x)
    return tf.nn.leaky_relu(x)

def PyramidPooling(inputs): # 
    #Dilated Spatial Pyramid Pooling) 
    
    dims = inputs.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1],
              dims[-2] // x.shape[2]),
        interpolation='bilinear'
    )(x)
    
    out1 = convolution_block(inputs, kernel_size=1, dilation_rate=1)
    out6 = convolution_block(inputs, kernel_size=3, dilation_rate=6)
    out12 = convolution_block(inputs, kernel_size=3, dilation_rate=12)
    out18 = convolution_block(inputs, kernel_size=3, dilation_rate=18)
    
    x = layers.concatenate([out_pool, out1, out6, out12, out18])
    
    return convolution_block(x, kernel_size=1)

def DeepLab(image_size):
    '''
    Create DNN with ResNet50 
    '''

    regularizer = tf.keras.regularizers.l1_l2(l1=0.001, l2=0.0001)
    
    inputs = layers.Input((*target_size, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_tensor=inputs
    )
    x = resnet50.get_layer('conv4_block6_2_relu').output
    x = PyramidPooling(x)
    
    input_a = layers.UpSampling2D(size=(target_size[0] // 4 // x.shape[1],
                                        target_size[1] // 4 // x.shape[2]),
                                 interpolation='bilinear')(x)
    
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=64, kernel_size=1, kernel_regularizer=regularizer)
    
    x = layers.concatenate([input_a, input_b])
    x = convolution_block(x, use_bias=True, kernel_regularizer=regularizer)
    x = convolution_block(x, use_bias=True, kernel_regularizer=regularizer)
    x = layers.UpSampling2D(size=(target_size[0] // x.shape[1],
                                  target_size[1] // x.shape[2]),
                           interpolation='bilinear')(x)

    outputs = layers.Conv2D(1, kernel_size=1, padding='same', activation=tf.nn.sigmoid)(x)
    
    return tf.keras.Model(inputs, outputs)

class GCAdam(tf.keras.optimizers.Adam): 
    '''
    #https://arxiv.org/abs/2004.01461
    Gradient Centralization: Optimization techniques are of great importance to effectively 
    and efficiently train a deep neural network (DNN). 
    '''
    def get_gradients(self, loss, params):
        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

def dice_coef(y_true, y_pred, smooth=1):
    '''
    Sorensen-Dyes metric to measure how similar two masks are
    '''
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return tf.reduce_mean((2. * intersection + smooth)/(union + smooth))

class Schedule():
    '''
    Decreasing training step schedule for a callback
    Decreasing starts at start_epoch and happens every per_epoch
    '''
    def __init__(self, start_epoch=5, per_epoch=1):
        self.start_epoch = start_epoch
        self.per_epoch = per_epoch

    def __call__(self, epoch, lr):
        if epoch >= self.start_epoch and epoch % self.per_epoch == 0:
            lr *= tf.math.exp(-0.1)
        print('learning rate:', float(lr))
        return lr

def weighted_binary_crossentropy(y_true, y_pred, weight=5.) :
    #weighted loss
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
    logloss = -(y_true * tf.keras.backend.log(y_pred) * weight + (1 - y_true) * tf.keras.backend.log(1 - y_pred))
    return tf.keras.backend.mean( logloss, axis=-1)

model = DeepLab(target_size)

# Train

epochs = 20 
loss = weighted_binary_crossentropy
optimizer = GCAdam(0.0002)
metrics = [tf.keras.metrics.BinaryAccuracy(),
           tf.metrics.MeanIoU(num_classes=2, name='IoU'), 
           dice_coef]

schedule_callback = tf.keras.callbacks.LearningRateScheduler(Schedule(10, 2))
#checkpoint for saving best model weights
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'airbus_weights.h5'), 
                                   monitor='val_loss', 
                                   verbose=1, 
                                   save_best_only=True,
                                   save_weights_only=True,
                                   mode='auto')
callbacks = [model_checkpoint,schedule_callback]

# model compilations
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    initial_epoch=0,
                    callbacks=callbacks
                    )

model.save(os.path.join(MODEL_DIR, 'airbus_model2.h5'), include_optimizer=False)