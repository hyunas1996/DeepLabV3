import os
import tempfile
import numpy as np
import tensorflow as tf
from PIL import ImageOps
from PIL import Image
from matplotlib import gridspec
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-type", required=True, help="image or video")
parser.add_argument("-file_path", required=True, help="file path")
#parser.add_argument("-model_type", required=True, help="dynamic, 16, 8")
args = parser.parse_args()

print(tf.__version__)


def load_model(tflite_path):
    #input = path of tflite model (coverted already)
    #output = model & model's desired input size

    #load TFLITE model & Get inpute image size
    
    print("####### STEP 1. LOAD MODEL #######")
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=tflite_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # get image size - converting from BHWC to WH
    input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
    print(input_size, end='\n\n')
    
    return interpreter, input_size



def crop_image(image, input_size):
    #input = input image & model's desired input size
    #output = cropped image for interpreter
    
    print("####### STEP 2. CROP IMAGE #######")
    
    old_size = image.size
    desired_ratio = input_size[0] / input_size[1]
    print("old size = ", old_size)
    old_ratio = old_size[0] / old_size[1]
    
    if old_ratio < desired_ratio: # '<': cropping, '>': padding
        new_size = (old_size[0], int(old_size[0] / desired_ratio))
    else:
        new_size = (int(old_size[1] * desired_ratio), old_size[1])

    print("new size : ", new_size)
    print("old size : ", old_size)
    
    # Cropping the original image to the desired aspect ratio
    delta_w = new_size[0] - old_size[0]
    delta_h = new_size[1] - old_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    cropped_image = ImageOps.expand(image, padding)

    print("croppped image's size : ", np.array(cropped_image).shape, end='\n\n')
    
    return cropped_image


def video_crop_frame(frame, input_size):
    #input = input image & model's desired input size
    #output = cropped image for interpreter
    
    print("####### STEP 2. CROP FRAME #######")
    
    old_size = frame.shape
    desired_ratio = input_size[0] / input_size[1]
    old_ratio = old_size[0] / old_size[1]
    
    if old_ratio < desired_ratio: # '<': cropping, '>': padding
        new_size = (old_size[0], int(old_size[0] / desired_ratio))
    else:
        new_size = (int(old_size[1] * desired_ratio), old_size[1])
    
    print("new size : ", new_size)
    print("old size : ", old_size)
    
    cropped_image = cv2.resize(frame, new_size)
    print("croppped image's size : ", np.array(cropped_image).shape, end='\n\n')
    
    return cropped_image


def resize_image(cropped_image, input_size):
    print("####### STEP 3. RESIZE IMAGE #######")
    
    # Resize the cropped image to the desired model size
    resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1

    print("resized image's size : ", np.array(resized_image).shape, end='\n\n')
    
    return image_for_prediction 

def video_resize_frame(cropped_image, input_size):
    print("####### STEP 3. RESIZE IMAGE #######")
    
    # Resize the cropped image to the desired model size
    resized_image = cv2.resize(cropped_image, input_size)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1

    print("resized image's size : ", np.array(resized_image).shape, end='\n\n')
    
    return image_for_prediction 

def inference(interpreter, image_for_prediction, cropped_image):
    
    print("####### STEP 4. INFERENCE #######")
    
    # Load the model.
    #interpreter = tf.lite.Interpreter(model_path=tflite_path)
    
    input_details = interpreter.get_input_details()
    
    # Invoke the interpreter to run inference.
    tmp_time = time.time()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    interpreter.invoke()

    # Retrieve the raw output map.
    raw_prediction = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()
    
    end_time = time.time()
    print("inference time : ",end_time - tmp_time)
    
    print("raw_prediction's size : ", raw_prediction.shape)

    # Post-processing: convert raw output to segmentation output
    ## Method 1: argmax before resize - this is used in some frozen graph
    # seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
    # seg_map = np.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
    ## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output
    width, height = cropped_image.size
    seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

    print("segmentation map's size : ", seg_map.shape, end='\n\n')
    
    return seg_map
    
def video_inference(interpreter, image_for_prediction, cropped_image):
    
    print("####### STEP 4. INFERENCE #######")
    
    # Load the model.
    #interpreter = tf.lite.Interpreter(model_path=tflite_path)
    
    input_details = interpreter.get_input_details()
    
    # Invoke the interpreter to run inference.
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    interpreter.invoke()

    # Retrieve the raw output map.
    raw_prediction = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()

    print("raw_prediction's size : ", raw_prediction.shape)

    # Post-processing: convert raw output to segmentation output
    ## Method 1: argmax before resize - this is used in some frozen graph
    # seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
    # seg_map = np.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
    ## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output
    width, height, _ = cropped_image.shape
    seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

    print("segmentation map's size : ", seg_map.shape, end='\n\n')
    
    return seg_map
    
    
#pascal related code

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
       Returns: A Colormap for visualizing segmentation results."""
    
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

      Args:
        label: A 2D array with integer type, storing the segmentation label.

      Returns:
        result: A 2D array with floating type. The element of the array
          is the color indexed by the corresponding element in the input label
          to the PASCAL color map.

      Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
          map maximum entry.
      """

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map, LABEL_NAMES):
    print("###### STEP 6. VISUALIZE ######")
    
    category_type = []

    category_set = np.unique(seg_map)

    print(category_set)
    
    print("category in this frame")
    for i in category_set:
        print(LABEL_NAMES[i])

    print()
    
    cropped_image_numpy = np.asarray(cropped_image, np.uint8)
    seg_image = seg_image = label_to_color_image(seg_map).astype(np.uint8)
    
    result_image = cv2.addWeighted(cropped_image_numpy, 0.4, seg_image, 0.6, 0)
    
    return result_image


if __name__ == '__main__':
    
    #tflite_path = 'lite-model_mobilenetv2-coco_dr_1.tflite'
    tflite_path = 'tmpg1or2mbe.tflite'
    dataset = 'pascal'
    
    interpreter, input_size = load_model(tflite_path)
    
    LABEL_NAMES = np.asarray(['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
        
    if args.type == 'image':
        print('input file type is IMAGE', end='\n\n')
    
        #image load
        image = Image.open(args.file_path)
        
        #image prepocessing
        cropped_image = crop_image(image, input_size) 
        image_for_prediction = resize_image(cropped_image, input_size) #(1, 513, 513, 3)
        
        #inference
        
        seg_map = inference(interpreter, image_for_prediction, cropped_image)
        
        #save results
        result_image = vis_segmentation(cropped_image, seg_map, LABEL_NAMES) 
        
        print('result image shape : ',result_image.shape)
        cv2.imwrite('result_image.jpeg', result_image)
        
        
    elif args.type == 'video':
        print('input file type is VIDEO')
        
        #video load
        video = cv2.VideoCapture(args.file_path)
        
        ret, frame = video.read()
        print("ret : ", ret)
        print("video frame size : ", frame.shape)
        
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
        print("width & height")
        print(width, height)
        print()
        
        fps = 30
        size = (int(width), int(height))
        pathOut = './bottle_test_result_pascal_xception.mp4'
        
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        
        total_inference_time = 0
        frame_num = 0
        while video.isOpened():
            
            ret, image = video.read()
            
            if not ret:
                break
            
            #image prepocessing
            print("video frame shape : ", image.shape) 
            cropped_image = video_crop_frame(image, input_size)
            image_for_prediction = video_resize_frame(cropped_image, input_size)
            
            #inference
            tmp_time = time.time()
            seg_map = video_inference(interpreter, image_for_prediction, cropped_image)
            total_inference_time += time.time() - tmp_time
            
            #save results
            result_image = vis_segmentation(cropped_image, seg_map, LABEL_NAMES)
            
            result_image = cv2.resize(result_image, size)
            
            out.write(result_image)
            frame_num += 1
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
          
        video.release()
        out.release
        
        print("average time for inference one frame is ", total_inference_time/frame_num)
