# Pascal-VOC
**_IIT-PKD: CS5007 - Deep Learning Project._**

### Members
- Yogesh R
- Subhash S
- Neeraj Patil

### General Info

- preprocessing.ipynb need to be executed first to get the cache, other ipynbs depend on it
- the cache folder holds all the important result of this project. Use the link below to access it.
    https://drive.google.com/drive/folders/1IOeSR6_gFxJtLrxPWzp1z6-memZZDpzL?usp=sharing
  
    The models are available in cache/models
- While loading the model make sure to have compile arg false and compile it after loading as a custom loss function is used.

### Problem Statement

This is a high level description which will naturally evolve. Perform multi-modal image classification and image description on PASCAL50S dataset. The PASCAL50S dataset is generated using the 1000 images taken from the PASCAL VOC 2008 dataset. The base dataset contains 6000 images belonging to 20 image classes. 50 images from each class are randomly sampled to form the 1000 images in PASCAL50S dataset[1][2]. Each image is annotated using 50 sentences, describing the image[1].

This is the link to download the dataset: https://filebox.ece.vt.edu/~vrama91/CIDEr_miscellanous/cider_datasets.tar


#### Basic Problem Statement
- The first task is to classify the images into 20 classes ( person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv/monitor). As the dataset is for image description tasks, you won't find the image labels off-the-shelf, but need to scrape it using the 50 sentence descriptions associated with each image. There might be cases where the exact class names are not present in the description. Think of an approach to overcome this issue. Also, some images may contain more than two objects belonging to different classes. So, basically treat this as a multi-label classification problem. Next, using the images and the scraped labels, perform simple image classification using deep convolutional neural networks. 
- In this task you will perform multi-modal image classification. Basically, you are expected to find a combined representation of text and images, which can be further used for image classification. You are free to adopt any technique for this purpose. You may need to perform a small literature review to know various techniques that can be used. 
- This is the final task. Using a combination of networks like RNN, CNN and probably FNN build a model for image description. The image and associated sentences can be used for training the model.


### Reference

[1]  Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. Collecting image annotations using amazon’s mechanical turk.  InProceedings  of  the  NAACL  HLT  2010  Workshop  on  CreatingSpeech and Language Data with Amazon’s Mechanical Turk, pages 139–147, 2010.

[2]  Ramakrishna  Vedantam,  C  Lawrence  Zitnick,  and  Devi  Parikh.   Cider:   Consensus-based  image description evaluation.  InProceedings of the IEEE conference on computer vision and pattern recog-nition, pages 4566–4575, 2015


