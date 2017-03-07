# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:28:08 2017

@author: Erik
"""
# library imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
import pickle
import os
from collections import deque

from proj_functions import build_file_list, data_look, find_cars, svc_classifier, add_heat, apply_threshold, draw_labeled_bboxes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





# Tweak these parameters and see how the results change.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

ystart = 400
ystop = 656
scale = 1.5
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Check if there is an existing trained SVC and X_scaler that can be used. If not, then train one, otherwise skip to the pipeline.
if not os.path.isfile("./svc.p") or not os.path.isfile("./X_scaler.p"):
            
    path_cars = './training_images/vehicles'
    path_not_cars = './training_images/non-vehicles'
    
    #build a list of all the training files
    car_images = build_file_list(path_cars)
    not_car_images = build_file_list(path_not_cars)
    
    cars = []
    not_cars = []
    
    for image in car_images:
        cars.append(image)
        
    for image in not_car_images:
        not_cars.append(image)
    
    #shuffle the order of the training data to avoid any patterns based on the order of the file being
    np.random.shuffle(cars)
    np.random.shuffle(not_cars)
    
    data_info = data_look(cars, not_cars)
    
    print('Your function returned a count of', 
          data_info["n_cars"], ' cars and', 
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', 
          data_info["data_type"])
    
    # train SVC using the car and not_car images with the parameters outlined at the top of the file
    svc, X_scaler = svc_classifier(cars, not_cars, color_space, spatial_size, hist_bins, hog_channel, orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat)
    
    # dump the trained SVC and X_scaler to a pickle file for future use.
    pickle.dump( svc, open( "svc.p", "wb" ) )
    pickle.dump( X_scaler, open( "X_scaler.p", "wb" ) )

# import trained SVC and corresponding X_scaler
svc = pickle.load(open( "svc.p", "rb" ))
X_scaler = pickle.load(open( "X_scaler.p", "rb" ))

#create a global heat_memory variable to keep track of the heatmap of car detections
heat_memory = deque([],maxlen = 3)

# image/video processing pipeline
def pipeline(input_img):
    global heat_memory
    current_heat = np.zeros_like(input_img[:,:,0]).astype(np.float)
    
    # build a list of bounding boxes that contain detected cars
    boxes = find_cars(input_img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    # Add heat to each box in boxes
    current_heat = add_heat(current_heat,boxes)
    
    # add the heat of detected cars to heat_memory and average out the contents of heat_memory so discount any one-off false positives 
    heat_memory.append(current_heat)
    heat = np.mean(heat_memory, axis=0)
    
    # Apply threshold to help remove false positives. Any detection that doesn't last for 3 frames is discarded
    thresh_heat = apply_threshold(heat,1)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(thresh_heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    output = draw_labeled_bboxes(np.copy(input_img), labels)
    
    #return image with bounding boxes drawn over the cars
    return output


# Code for processing test images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_img = mpimg.imread('./test_images/test6.jpg')

result = pipeline(test_img)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

ax1.imshow(test_img)
ax1.axis("off");

ax2.imshow(result)
ax2.axis("off");
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# Code for processing video stream
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from moviepy.editor import VideoFileClip

project_output = 'output_images/project.mp4'
clip1 = VideoFileClip('project_video.mp4', audio=False)
project_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
project_clip.preview()
project_clip.write_videofile(project_output, audio=False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~