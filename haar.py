#get the features 
import cv2
import numpy as np
import pandas as pd
import skimage

from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import argparse as ap
import glob
import os


#model testing
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import cv2

clf = joblib.load('models/dump2')

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
def overlapping_area(detection_1, detection_2):
    # Calculate the x-y co-ordinates of the 
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=.5):
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
            reverse=True)
    # Unique detections will be appended to this list
    new_detections=[]
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections

import warnings
warnings.filterwarnings('ignore')
import time


filets=[]
for i in glob.glob("Test/*.jpg"):
    filets.append(i)

ans=[]
detectors=[]
f = open('test.txt','a+')
for ima in filets :
    im = imread(ima, as_grey=True)
    min_wdw_sz = (63, 57)
    step_size = (10, 10)
    downscale = 1.5
    visualize_det =False

    # clf = joblib.load()
    detections = []
    # The current scale of the image
    scale = 0
    cut=0
    start = time.time()
    print("hello")
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # detections at the current scale
        cd = []
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # Calculate the HOG features
            fd,hog_image2 = hog(im_window,8,(5,5),(1,1), visualise=True)
    #         print(fd.50shape)
            pred = clf.predict(fd)
            if pred == 1:
                print ("Detection:: Location -> ({}, {})".format(x, y))
                print ("Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd)))
                detections.append((x, y, clf.decision_function(fd),
                    int(min_wdw_sz[0]*(downscale**scale)),
                    int(min_wdw_sz[1]*(downscale**scale))))
                cd.append(detections[-1])
#             if visualize_det:
#                 clone = im_scaled.copy()
#                 for x1, y1, _, _, _  in cd:
#                     cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
#                         im_window.shape[0]), (0, 0, 0), thickness=2)
#                 cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
#                     im_window.shape[0]), (255, 255, 255), thickness=2)

    #             cv2.imwrite('bound/'+str(cut)+'.jpg',clone)
    #             cut+=1
    #             cv2.waitKey(30)
        # Move the the next scale
        scale+=1

    # Display the results before performing NMS
    clone = im.copy()
#     for (x_tl, y_tl, _, w, h) in detections:
#         # Draw the detections
#         cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
#     cv2.imwrite('bound/'+str(1)+'.jpg', im)
    end = time.time()
    print(end - start)
    print(ima)
    print (detections)
    p=nms(detections)
    detectors.append((ima,p))
    ans.append((ima,p[0]))
    
    f.write(ima)
    f.write(p[0])
    f.write("\n")
f.close()