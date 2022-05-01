# Project Report
*Authors: Weicheng Dai, Yunyi Zhu, Hansheng Li*

## 1. Abstract  
Nowadays detecting similar images has been very useful in many cases, such as human identification, human tracking. One way to solve this problem is to find the most similar person in an image database that is already achieved. In this work, we propose a pipeline to process the image database, and use it to find similar people when given a new image. We have adopted a face alignment algorithm, an embedding algorithm and k-nearest neighbor finding algorithm to provide the result. The result was salient, in that we could return a bunch of similar images.

## 2. Introduction  
Face alignment and search are essential parts in many face recognition engineering applications. We can imagine the significance of facial expression analysis and human identification or tracking. However, in many cases, we are given images which contain many noises, such as other parts of humans, or other background noises, and this would no doubt impose great challenges in this mission. In this paper, we propose a framework for finding faces in an image, and get semantic representations, then retrieve the most similar faces in the dataset. The result shows that given certain images containing faces, our pipeline can effectively return the nearest faces in the database.

## 3. Related Work
Human face alignment   
There are other neural networks that deal with detection, such as YOLO, R-CNN and fast R-CNN. However, these neural networks mainly deal with the detection of normal objects, such as sports balls, computers, or cars. In this work, we have to align human faces, which is different from the job of these networks because only human faces are what we need. MTCNN is widely used in videos while human faces are being tracked. We will talk about why MTCNN is an essential part of this job in the following sections. In reality, in this experiment, MTCNN is pre-trained with the Deepfake Detection Challenge.   

Human face feature extraction  
Many other transfer learning uses simple neural networks such as VGGNet or LeNet, but here in the baseline part we tried to use ResNet-50 which should run much faster and deeper. In the improvement part, we are instead using Inception Resnet to better enhance the performance. Also, facenet is a widely used approach of dealing with human faces, and it uses triplet loss for training. However, in this experiment, since most of the images in the dataset don’t have their corresponding positive samples, we are instead using contrastive loss that only compares positive and negative samples. 

## 4. Data
The data we are using is [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?select=matchpairsDevTrain.csv) (Labeled Faces in the Wild). It contains 13233 images of 5749 people.  
The LFW dataset has following characteristics:  
1. It contains extra information other than faces. As we can see in figure 1(Aaron Patterson and Aaron Guiel), there are some grids that are shown in black. Also, the right image shows hands of both people, which is not part of the task of identifying human faces. Such characteristics will no doubt greatly enhance the difficulty of finding human faces inside the database.  
2. Imbalanced distribution of images. In the LFW dataset, 1680 people have two or more images in the database. The remaining 4069 people have just a single image in the database. If we deal with this problem using the traditional classification method, the results will no doubt be biased, meaning that the feature space vectors will be close to those labels with larger numbers of faces in the dataset.   
3. Diversity. The LFW dataset contains many ethnicities, which is abundant enough for us to conduct this research.  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure1.png)
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure2.png)     

At first, we did not apply any noise filtering method, and view this problem as a classification problem. That naturally leads to the problem of biased results. Therefore, after considering that there are many noises in the dataset, we tried to first apply an algorithm to align the faces in the dataset. The model that we use in the preprocessing part is [MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) (Multi-Task Cascaded Convolutional Networks). Basically we use MTCNN to first align the faces in the images, and then process the image. In figure2, which corresponds to figure 1, we can see that only faces are detected and kept for future use. In this way, backgrounds and noises are ignored, which is better for our training session.  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure3.png)
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure4.png)

## 5. Methods
The main pipeline is shown in the figure below. Basically we use two models for our job. One is the MTCNN that preprocesses the image and gets the face that we want. And the other is an InceptionResNet with a decoder to get the embeddings of facial images and rebuild them with embeddings. The reason for utilizing a decoder is that we want to avoid the collapsed solution of getting the embeddings. The contrastive loss between embeddings is shown below:
$$l_{contra} = \frac{1}{2} \times (\tau \times dis +(1+(-1) \times \tau) \times ReLu(mar-\sqrt {dis})^2)$$

In this equation, $\tau$ is the target of two embeddings: 1 means they are from images of the same person, 0 otherwise. $dis$ means the euclidean distance between two embeddings. $mar$ is the margin between clusters. As we can see from the equation, when $\tau$ = 0, it means they are not from the same person, then we have to increase the distance of two embeddings, in which case we want the distance to be as large as possible. When $\tau$ =1, then we only compute the $\tau$ &times; $dis$, which means we want to minimize the distance.  
 The consistency loss is mean square error loss between the reconstructed image and the original one. We adopt this part to ensure that the embedding vector contains all the essential features in the original image. 
 ![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure5.png)  

MTCNN  
MTCNN contains three different Convolutional neural networks, with different missions. They are P-Net, R-Net and O-Net. The following paragraphs will introduce each of them and their functions.  
The first one is P-Net, which stands for Proposed net. P-Net is a convolutional neural network, and after extracting features, it generates two branches, one for classification (with channel 2), and the other for bounding boxes (with channel 4). 2 channels stands for the possibility of whether this is the object that we want to retrieve, basically we have to pass this information to a softmax layer. 4 channels stands for the coordinates of both upper left and lower right.  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure6.png)  

The Proposed net tries to find the bounding boxes of objects in a given image. This would be extremely useful to locate possible faces. In reality, images are scaled by different factors to generate several sizes of the same image, which we call an image pyramid. And then we feed this image pyramid to the Proposed net, aiming to better detect the faces of different sizes. As we can visualize from the figure below, the bounding boxes are the outputs of P-Net. Many objects, including some of the human faces, are detected, yet many of them are tracked by several bounding boxes.  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure7.png)  

However, since bounding boxes may overlap, we have to use non-maximum suppression to decrease the number of overlapping bounding boxes. The regression part of the P-Net, basically the 4 channel output, is also used in this part to assist calibration. Non-maximum suppression is a widely used technique in computer vision. Canny’s edge detection, Regional-CNN all use this for getting the final bounding. After using the Non-maximum suppression, bounding boxes would be reduced to the significant ones with valid coordinates and high possibilities.  
The second one is R-Net, which stands for Refine net. The Refine net is a further step of limiting the positive samples. It has a structure like the figure shown below. Similar to P-Net, it is also a convolutional neural network, but the difference is that the two branches at last are fully connected layers. Same as above, 2 means this is headed for classification, and 4 means that it is for bounding box regression.
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure8.png)  

After the Refine net, the large numbers of bounding boxes will be reduced to only those reasonable ones, since the Refine net takes the bounding boxes aligned images of P-Net, resizes them and applies itself in smaller regions. The Refine net would also give us the probabilities of whether a region denoted by a bounding box contains the object that we want, in this case, face. The example which shows the output of Refine net is shown below:  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure9.png) 

As we can see, the number of bounding boxes is greatly reduced, and most of them are aligned to the human faces, which is what we desire. Though, some problems, such as several bounding boxes denote the same face, still exist. That is why we need to use the last neural network.  
The third network is O-Net, which stands for Output net. The Output net has a similar structure as the R-Net, except that the input image size is doubled and a landmark of length 10 is returned. In an image of a human face, there are five points that are called landmarks, which are both eyes, nose, and both corners of mouth. Since each of them is denoted by 2 coordinates, we need 10 numbers in all. Same function of the other two, as the ones mentioned above, is used for classification and bounding box regression.
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure10.png) 

After using the Output net, human faces with the landmarks are shown, as is illustrated in the figure below. With this method, Our network can detect and locate each human face step by step, and ensure that no details are missed in this procedure, and no redundant outputs.   
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure11.png) 

Inception Resnet  
Inception Resnet, also known as the GoogLeNet, is based on the Resnet, which is already covered in the lecture. Inception means that this model contains several different blocks in one module, as opposed to the VGGNet or LeNet. The detailed structure of Inception Resnet can be found on the 7th page of the original paper. Basically instead of using a single loop, multiple heads of inference would generate much deeper features in one module. However, this naturally leads to the problem that it takes more iterations to train such a network. At last, the Inception Resnet returns a feature of 512 dimensions for each input image, which we are going to be using for the last procedure of finding the k-nearest neighbors for each query image.  
Decoder  
Using an Inception Resnet is sufficient to accomplish the mission of embedding images, but we raise a question: Could there be a possibility that such kind of encoder would memorize each image and thus produce collapsed features? To avoid this from happening, an auxiliary decoder is added after extracting the feature. Basically the decoder is a MLP that maps the feature into image space using transposed convolutional layers. The reconstructed images are then used to compare with original images, in that we want the features to include all the essential components of original images. It turns out that with this method, the problem of collapsed features could be solved.   

## 6. Experiments
Reverse Image Search Baseline  
To achieve the baseline of Reverse Image Search, we first can do a transfer learning by using a pretrained model to train the LWF Dataset and get a new model. In this project we have chosen ResNet-50 as a pretrained model and using PyTorch to implement the function. First, we need to normalize the tensor image with mean and standard deviation, resize the image to 224 * 224, shuffled  training dataset, and set batch size to 64. Then build two fully connected layers with 5749 nodes in the end, since we have 5749 people in the data set and use a rectified linear unit as the activate function. After that we use CrossEntropy to calculate the loss so we can do the backpropagation and use Adam as the optimizer to improve the model. By setting 100 epochs, as the result turns out we can see the loss becomes smaller and accutricity is higher after each epoch. Then save the model named weights.h5.  
Now we have a model with good accuricity, next step is to create a feature vector space to store all the feature vectors for each image. So that we can use the k-nearest neighbors algorithm to calculate Euclidean distance between the target's feature vector calculated by running though the new model with the feature vector space, the smaller the  Euclidean distance is, the pictures are similar to the target image.  
The result of the nearest 20 images, respect to the 10 query images  
As you can see the result is not perfect, but still can find some correlation. There are many reasons for this, the main reason is the training data set is very unbalanced, data is not enough, or maybe the training model is not deep enough.  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure12.png)
<img src="https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure12.png" width="600"  >
Reverse Image Search Improvement  
In Reverse Image Search Improvement, the MTCNN was formerly trained with the Deepfake Detection Challenge and reloaded. In Inception Resnet part, the loss function is computed as this:$loss = l_{contra} + l_{consist}$, where $l_{contra}$ is represented by equation above, and $l_{consist}$ is the MSE loss between the generated image and the original image. We trained for 100 epochs with an initialized learning rate of 1e-3, then divided by 10 in epoch 10, 20 and 50. Whole training session takes about 10 hours.  
The result of the nearest 20 images in database, respect to the 10 query images are shown as below:  
![image](https://github.com/Joey-99/Reverse-Visual-Search/blob/main/docs/img%20files/figure13.png) 

## 7. Conclusion
The result of reverse image search was salient, in that we can successfully retrieve the nearest neighbors in the embeddings of images. We have learnt the pipeline of preprocessing datasets, creating neural networks using pytorch, and finding useful loss functions that can better enhance the performance. Also, to use the knn search, we can use some useful tools in pytorch to compute the distance, which is very convenient.  
For future use, maybe we can use more datasets to train the model, since the LFW dataset is actually highly imbalanced. Combining with other dataset would help a lot. We can thus build a larger database of embeddings and thus return more plausible results.  

## 8. Extra credit: Reverse Video Search 
Video can be seen as a combination of multiple frames of images. The experiment is divided into 3 steps in total. Firstly, cutting the video and each frame is cut into one image. Then, the image is fed into the feature detector. Three algorithms for calculating image similarity are used here.  
1. ORB algorithm(Oriented FAST and Rotated BRIEF)  
ORB can be used to quickly create feature vectors of key points in an image, which can be used to identify objects in the image. It can be divided into fast and brief parts which are the feature detection algorithm and vector creation algorithm, respectively.   
ORB first looks for special regions, called keypoints, from the image. Keypoints are small areas of the image that stand out, such as corner points, for example, they have the characteristic of dramatically changing pixel values from light to dark. ORB then calculates the corresponding feature vector for each key point.  
The feature vector created only contains 0 and 1, called a binary feature vector. The order of 1s and 0s varies depending on the particular keypoint and the region of pixels surrounding it. The vector represents the intensity pattern around the key point, so multiple feature vectors can be used to identify larger areas or even specific objects in the image.  
ORB is characterized by being ultra-fast and somewhat independent of noise and image transformations, such as rotation and scaling transformations.  
2. phash algorithm  
First of all, unify the images into the same specifications, in this experiment we used a resolution of 256*256. Then, the images are grayed out at the pixel level. And apply DCT (discrete cosine transform) compression algorithm to the images. Then calculate  DCT mean value and hash value and compare each DCT value with the average value. If greater than or equal to the mean value, recorded as 1. Otherwise record as 0 and this generates a binary array. Finally, the pairing of images is performed and the Hamming distance is calculated to determine the similarity.  
3. Histogram matching  
Resize the image to get the same size image. Grayscale the image, the pixels of the image after grayscale are between 0-255. Calculate the histogram data of the image, count the probability distribution of the same pixel, and calculate the correlation of the histogram of two images.  
The image histogram is rich in image detail information and reflects the probability distribution of image pixel points, counting the number of pixels each pixel point intensity value has. It is relatively small in computational effort. However, the histogram reflects the probability distribution of image gray value and does not have the spatial location information of the image in it, so it can be misjudged. For example, images with the same texture structure but different light and darkness should have high similarity, but the actual result is low similarity, while images with different texture structure but similar light and darkness have high similarity.  

Finally we determine the final similarity by comparing the similarity of the three algorithms and setting the threshold.

## References
1. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
2. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
3. Going deeper with convolutions
4. https://github.com/TreB1eN/InsightFace_Pytorch.git
5. https://github.com/timesler/facenet-pytorch.git

