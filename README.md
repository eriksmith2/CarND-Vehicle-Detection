# Vehicle Detection Project

### Project Goals:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
2. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
3. Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
4. Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
5. Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
6. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./test_images/test6.jpg
[image2]: ./output_images/data_sample.png
[image3]: ./output_images/HOG.png
[image4]: ./output_images/box_img.png
[image5]: ./output_images/current_heat.png
[image6]: ./output_images/thresh_heat.png
[image7]: ./output_images/output.png
[video1]: ./output_images/project.mp4

---
### Project:

### Histogram of Oriented Gradients (HOG)

I started by reading in all the `car` and `not-car` images.  Here is an example of one of each of the `car` and `not-car` classes:

![alt text][image2]

There are roughly 8500 of each type of image in the set of training images. I read in all of the images using `build_file_list` and created a list of `cars_images` and `not_car_images`:

```python
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
```
```python
# find all files of a specified file type within a folder and its subfolders
def build_file_list(path, file_type = '.png'):
    files = [os.path.join(root, name)
        for root, dirs, files in os.walk(path)
        for name in files
        if name.endswith((file_type))]
    return files
```

I then shuffled each list to improve the accuracy by preventing patterns in the order of the files from influencing the training of my classifier.

Using trial and error, I tweaked the parameters governing the extraction of HOG and color features until I came across a combination that seemed to yield good results in the test accuracy of my classifier:

```python
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
```

These HOG parameters result in images like this from each channel of YCrCb color space input images:

![alt text][image3]

With the above parameters, I was able to extract HOG and color features and train an SVC classifier using the training set of images. Both the extraction, and training were done in my `svc_classifier` function:

```python
# train a SVC classifier using input arrays of labeled training images. return the trained SVC and the scaler used.
def svc_classifier(cars, not_cars, colorspace, spatial_size, hist_bins, hog_channel, orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat):

    sample_size = 8500
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]

    t=time.time()
    car_features = extract_features(cars, colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

    not_car_features = extract_features(not_cars, colorspace, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler
```
I created an `extract_features` function that takes the HOG and color parameters and returns a list of features:

```python
# Extract features from a list of images. code from Udacity lectures
def extract_features(imgs, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
```

The `svc_classifier` also splits the training images into a training set (80%) and a testing set (20%) and, after training, returns the calculated accuracy of the classifier. With the parameters above the calculated accuracy of the classifier is approximately 99% on the test set.


### Sliding Window Search

In order to find cars in test images, and ultimately the video, a sliding window search was used. The idea is to split the input image into smaller windows that can be analyzed and any windows where a car is detected are added to a list of good windows. The parameters that govern the scale and the overlap of the windows were tuned through trial and error on the test images until the search was reliably finding the cars in the image. I found that by using a combination of all three color channel's HOG features, as well as the color features (both spatial and histogram) I was able to improve the reliability of the classifier.  A sample output image and the code are shown below:

![alt text][image4]

```python
# function that can extract features using hog sub-sampling and make predictions. code from Udacity lectures
def find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = img_tosearch
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    #nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Initialize a list to append detected positions to
    boxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(hog_features.reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

    return boxes
```
Rather than directly outputting an image, the `find_cars` function returns a list of all the windows in which cars were detected. This list of windows can then be processed further in order to eliminate false positives and generally clean up the pipeline output.

In the pipeline, the list of windows is refined into a single overall bounding box through heatmapping. Basically, all pixels contained within each window in the list of windows output by `find_cars` have a 'heat' value assigned to them. Each pixel inside a bounding box has it's heat value increased by 1; if a pixel lies in one box only it has a value of 1, if it is inside two boxes, it gets a value of 2 and so on. The output image of this process looks like this:

![alt text][image5]

This process is completed in the `add_heat` function:

```python
# add heat to pixels inside the input list of bounding boxes. code from Udacity lectures
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
```

Since this heatmap is built from easy to work with integer values for each pixel, it is easy to threshold in order to eliminate false positives and detections of low certainty. using the `apply_threshold` function, I am able to eliminate any detections that are in, for example, only one window. By thresholding away one off detections, false positives are reliably removed from the output.

```python
# threshold input heatmap. code from Udacity lectures
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```
![alt text][image6]

Finally, with the thresholded heatmap, new bounding boxes can be drawn around any areas where the heat value exceeds a given threshold. I did this using the `draw_labeled_bboxes` function.

```python
# Draw bounding boxes around pixels with value >1. code from Udacity lectures
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

This function returns the output image of my pipeline:

![alt text][image7]

---

### Video Implementation

My pipeline works in roughly the same way on videos as it does on one off images. The main difference is that the video pipeline takes the average heat value over three frames and uses that value to smooth out the normally jittery results and further reduce the chance of false positive detections making it through to the output.

Here's a [link to my video result](./output_images/project.mp4)

---

### Discussion

The biggest remaining issues in my pipeline are that the initial detection of a vehicle entering the right side of the screen takes longer than it should and the detection algorithm seems to lose track of the white vehicle when it gets too far away from the camera. My theory as to why this occurs is that the training data I used may not have enough example images of cars in these situations, so the pipeline fails to recognize them as positive detections. The easiest way to resolve these issues would be to grab the specific frames where the detection algorithm fails, and use those images to refine the training of the classifier. The difficulty then lies in ensuring that the new training is still generalized and not functional only on the specific video I am testing.

One other improvement would be to further fine tune the parameters used in extracting features from the images fed into the training and detection pipelines. Since all parameters were tuned primarily through trial and error, I'm sure there exists a combination that would further improve the results and the time to tune and test results is the only real constraint against improving the algorithm.
