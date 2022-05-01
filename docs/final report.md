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
## 6. Experiments

## 7. Conclusion