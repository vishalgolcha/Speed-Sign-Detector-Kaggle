# Speed-Sign-Detector-Kaggle-
Predict the Bounding Boxes of Speed Signs from Scene StreetView Images.The dataset was compiled by Map My India.
![Sample Image](https://github.com/vishalgolcha/Speed-Sign-Detector-Kaggle-/blob/master/Images/sample.jpg)
> Task at hand was to detect bounding boxes around speed signs in a small dataset given by Map My India . A sample image has been shown above .
> - Techniques used were Histogram of Gradients(HOG) features  to detect the circular shaped signboards .
> - To detect bounding boxes accurately a sliding window algorithm was used , which gave predictions as confidence scores .
> - A sample of the extracted hog features from a sign board : ![sample_hog_features](https://github.com/vishalgolcha/Speed-Sign-Detector-Kaggle-/blob/master/Images/hog2.png)
> - The complete model is described in the jupyter notebook floydjup.ipynb
> - several dump named files have parameters for trained models.
