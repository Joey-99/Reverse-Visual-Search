<!-- Output copied to clipboard! -->

<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see inline alerts.
* ERRORs: 0
* WARNINGs: 2
* ALERTS: 22

Conversion time: 1.44 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β33
* Sat Apr 30 2022 11:18:51 GMT-0700 (PDT)
* Source doc: AI-Report

WARNING:
You have some equations: look for ">>>>>  gd2md-html alert:  equation..." in output.


WARNING:
Inline drawings not supported: look for ">>>>>  gd2md-html alert:  inline drawings..." in output.

* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 2; ALERTS: 22.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>
<a href="#gdcalert4">alert4</a>
<a href="#gdcalert5">alert5</a>
<a href="#gdcalert6">alert6</a>
<a href="#gdcalert7">alert7</a>
<a href="#gdcalert8">alert8</a>
<a href="#gdcalert9">alert9</a>
<a href="#gdcalert10">alert10</a>
<a href="#gdcalert11">alert11</a>
<a href="#gdcalert12">alert12</a>
<a href="#gdcalert13">alert13</a>
<a href="#gdcalert14">alert14</a>
<a href="#gdcalert15">alert15</a>
<a href="#gdcalert16">alert16</a>
<a href="#gdcalert17">alert17</a>
<a href="#gdcalert18">alert18</a>
<a href="#gdcalert19">alert19</a>
<a href="#gdcalert20">alert20</a>
<a href="#gdcalert21">alert21</a>
<a href="#gdcalert22">alert22</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>


Project Proposal:  

Group 8

Reverse-Visual-Search with LFW, Weicheng Dai, Yunyi Zhu, Hansheng Li



1. <span style="text-decoration:underline;">Abstract: Briefly describe your problem, approach, and key results. Should be no more than 300 words.</span>

    Nowadays detecting similar images has been very useful in many cases, such as human identification, human tracking. One way to solve this problem is to find the most similar person in an image dataset that is already achieved. In this work, we propose a pipeline to process the image database, and use it to find similar people when given a new image. We have adopted a face alignment algorithm, an embedding algorithm and k-nearest neighbor finding algorithm. The result was salient, in that we could return a bunch of similar images.

2. <span style="text-decoration:underline;">Introduction (10%): Describe the problem you are working on, why it’s important, and an overview of your results</span>

    Face alignment and search are essential parts in many face recognition engineering applications. We can imagine the significance of facial expression analysis and human identification or tracking. However, in many cases, we are given images which contain many noises, such as other parts of humans, or other background noises, and this would no doubt impose great challenges in this mission. In this paper, we propose a framework for finding faces in an image, and get semantic representations, then retrieve the most similar faces in the dataset. The result shows that given certain images containing faces, our pipeline can effectively return the nearest faces in the database.

3. <span style="text-decoration:underline;">Related Work (10%): Discuss published work that relates to your project. How is your approach similar or different from others?</span>

    Human face alignment 


    There are other neural networks that deal with detection, such as YOLO, R-CNN and fast R-CNN. However, these neural networks mainly deal with the detection of normal objects, such as sports balls, computers, or cars. In this work, we have to align human faces, which is different from the job of these networks because only human faces are what we need. MTCNN is widely used in videos while human faces are being tracked. We will talk about why MTCNN is an essential part of this job in the following sections. In reality, in this experiment, MTCNN is pre-trained with the Deepfake Detection Challenge. 


    Human face feature extraction


    Many other transfer learning uses simple neural networks such as VGGNet or LeNet, but here in the baseline part we tried to use ResNet-50 which should run much faster and deeper. In the improvement part, we are instead using Inception Resnet to better enhance the performance. Also, facenet is a widely used approach of dealing with human faces, and it uses triplet loss for training. However, in this experiment, since most of the images in the dataset don’t have their corresponding positive samples, we are instead using contrastive loss that only compares positive and negative samples. 

4. <span style="text-decoration:underline;">Data (10%): Describe the data you are working with for your project. What type of data is it? Where did it come from? How much data are you working with? Did you have to do any preprocessing, filtering, or other special treatment to use this data in your project?</span>

    The data we are using is [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?select=matchpairsDevTrain.csv) (Labeled Faces in the Wild). It contains 13233 images of 5749 people. 


    The LFW dataset has following characteristics:

1. It contains extra information other than faces. As we can see in figure 1(Aaron Patterson and Aaron Guiel), there are some grids that are shown in black. Also, the right image shows hands of both people, which is not part of the task of identifying human faces. Such characteristics will no doubt greatly enhance the difficulty of finding human faces inside the database.
2. Imbalanced distribution of images. In the LFW dataset, 1680 people have two or more images in the database. The remaining 4069 people have just a single image in the database. If we deal with this problem using the traditional classification method, the results will no doubt be biased, meaning that the feature space vectors will be close to those labels with larger numbers of faces in the dataset. 
3. Diversity. The LFW dataset contains many ethnicities, which is abundant enough for us to conduct this research.

    

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")



    At first, we did not apply any noise filtering method, and view this problem as a classification problem. That naturally leads to the problem of biased results. Therefore, after considering that there are many noises in the dataset, we tried to first apply an algorithm to align the faces in the dataset. The model that we use in the preprocessing part is [MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) (Multi-Task Cascaded Convolutional Networks). Basically we use MTCNN to first align the faces in the images, and then process the image. In figure2, which corresponds to figure 1, we can see that only faces are detected and kept for future use. In this way, backgrounds and noises are ignored, which is better for our training session.


    

<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")


<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")


5. <span style="text-decoration:underline;">Methods (30%): Discuss your approach for solving the problems that you set up in the introduction. Why is your approach the right thing to do? Did you consider alternative approaches? You should demonstrate that you have applied ideas and skills built up during the quarter to tackling your problem of choice. It may be helpful to include figures, diagrams, or tables to describe your method or compare it with other methods.</span>

    The main pipeline is shown in the figure below. Basically we use two models for our job. One is the MTCNN that preprocesses the image and gets the face that we want. And the other is an InceptionResNet with a decoder to get the embeddings of facial images and rebuild them with embeddings. The reason for utilizing a decoder is that we want to avoid the collapsed solution of getting the embeddings. The contrastive loss between embeddings is shown below:


    

<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>




    In this equation, 

<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 is the target of two embeddings: 1 means they are from images of the same person, 0 otherwise. 

<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 means the euclidean distance between two embeddings. 

<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 is the margin between clusters. As we can see from the equation, when 

<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

, it means they are not from the same person, then we have to increase the distance of two embeddings, in which case we want the distance to be as large as possible. When 

<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

, then we only compute the 

<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

, which means we want to minimize the distance. 


    The consistency loss is mean square error loss between the reconstructed image and the original one. We adopt this part to ensure that the embedding vector contains all the essential features in the original image. 


    

<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)


    MTCNN


    MTCNN contains three different Convolutional neural networks, with different missions. They are P-Net, R-Net and O-Net. The following paragraphs will introduce each of them and their functions.


    The first one is P-Net, which stands for Proposed net. P-Net is a convolutional neural network, and after extracting features, it generates two branches, one for classification (with channel 2), and the other for bounding boxes (with channel 4). 2 channels stands for the possibility of whether this is the object that we want to retrieve, basically we have to pass this information to a softmax layer. 4 channels stands for the coordinates of both upper left and lower right.


    

<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)


    The Proposed net tries to find the bounding boxes of objects in a given image. This would be extremely useful to locate possible faces. In reality, images are scaled by different factors to generate several sizes of the same image, which we call an image pyramid. And then we feed this image pyramid to the Proposed net, aiming to better detect the faces of different sizes. As we can visualize from the figure below, the bounding boxes are the outputs of P-Net. Many objects, including some of the human faces, are detected, yet many of them are tracked by several bounding boxes.


    

<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")



    However, since bounding boxes may overlap, we have to use non-maximum suppression to decrease the number of overlapping bounding boxes. The regression part of the P-Net, basically the 4 channel output, is also used in this part to assist calibration. Non-maximum suppression is a widely used technique in computer vision. Canny’s edge detection, Regional-CNN all use this for getting the final bounding. After using the Non-maximum suppression, bounding boxes would be reduced to the significant ones with valid coordinates and high possibilities.


    The second one is R-Net, which stands for Refine net. The Refine net is a further step of limiting the positive samples. It has a structure like the figure shown below. Similar to P-Net, it is also a convolutional neural network, but the difference is that the two branches at last are fully connected layers. Same as above, 2 means this is headed for classification, and 4 means that it is for bounding box regression.


    

<p id="gdcalert15" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert16">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)


    After the Refine net, the large numbers of bounding boxes will be reduced to only those reasonable ones, since the Refine net takes the bounding boxes aligned images of P-Net, resizes them and applies itself in smaller regions. The Refine net would also give us the probabilities of whether a region denoted by a bounding box contains the object that we want, in this case, face. The example which shows the output of Refine net is shown below:


    

<p id="gdcalert16" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert17">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")



    As we can see, the number of bounding boxes is greatly reduced, and most of them are aligned to the human faces, which is what we desire. Though, some problems, such as several bounding boxes denote the same face, still exist. That is why we need to use the last neural network.


    The third network is O-Net, which stands for Output net. The Output net has a similar structure as the R-Net, except that the input image size is doubled and a landmark of length 10 is returned. In an image of a human face, there are five points that are called landmarks, which are both eyes, nose, and both corners of mouth. Since each of them is denoted by 2 coordinates, we need 10 numbers in all. Same function of the other two, as the ones mentioned above, is used for classification and bounding box regression.


    

<p id="gdcalert17" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert18">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)


    After using the Output net, human faces with the landmarks are shown, as is illustrated in the figure below. With this method, Our network can detect and locate each human face step by step, and ensure that no details are missed in this procedure, and no redundant outputs. 


    

<p id="gdcalert18" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert19">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")



    Inception Resnet


    Inception Resnet, also known as the GoogLeNet, is based on the Resnet, which is already covered in the lecture. Inception means that this model contains several different blocks in one module, as opposed to the VGGNet or LeNet. The detailed structure of Inception Resnet can be found on the 7th page of the original paper. Basically instead of using a single loop, multiple heads of inference would generate much deeper features in one module. However, this naturally leads to the problem that it takes more iterations to train such a network. At last, the Inception Resnet returns a feature of 512 dimensions for each input image, which we are going to be using for the last procedure of finding the k-nearest neighbors for each query image.


    Decoder


    Using an Inception Resnet is sufficient to accomplish the mission of embedding images, but we raise a question: Could there be a possibility that such kind of encoder would memorize each image and thus produce collapsed features? To avoid this from happening, an auxiliary decoder is added after extracting the feature. Basically the decoder is a MLP that maps the feature into image space using transposed convolutional layers. The reconstructed images are then used to compare with original images, in that we want the features to include all the essential components of original images. It turns out that with this method, the problem of collapsed features could be solved. 

6. <span style="text-decoration:underline;">Experiments (30%): Discuss the experiments that you performed to demonstrate that your approach solves the problem. The exact experiments will vary depending on the project, but you might compare with previously published methods, perform an ablation study to determine the impact of various components of your system, experiment with different hyperparameters or architectural choices, use visualization techniques to gain insight into how your model works, discuss common failure modes of your model, etc. You should include graphs, tables, or other figures to illustrate your experimental results.</span>

    To achieve the baseline of Reverse Image Search, we first can do a transfer learning by using a pretrained model to train the LWF Dataset and get a new model. In this project we have chosen ResNet-50 as a pretrained model and using PyTorch to implement the function. First, we need to normalize the tensor image with mean and standard deviation, resize the image to 224 * 224, shuffled  training dataset, and set batch size to 64. Then build two fully connected layers with 5749 nodes in the end, since we have 5749 people in the data set and use a rectified linear unit as the activate function. After that we use CrossEntropy to calculate the loss so we can do the backpropagation and use Adam as the optimizer to improve the model. By setting 100 epochs, as the result turns out we can see the loss becomes smaller and accutricity is higher after each epoch. Then save the model named weights.h5.


    Now we have a model with good accuricity, next step is to create a feature vector space to store all the feature vectors for each image. So that we can use the k-nearest neighbors algorithm to calculate Euclidean distance between the target's feature vector calculated by running though the new model with the feature vector space, the smaller the  Euclidean distance is, the pictures are similar to the target image.


    Reverse Image Search Improvement


    In Reverse Image Search Improvement, the MTCNN was formerly trained with the Deepfake Detection Challenge and reloaded. In Inception Resnet part, the loss function is computed as this:

<p id="gdcalert19" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert20">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

, where 

<p id="gdcalert20" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert21">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

is represented by equation above, and 

<p id="gdcalert21" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert22">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 is the MSE loss between the generated image and the original image. We trained for 100 epochs with an initialized learning rate of 1e-3, then divided by 10 in epoch 10, 20 and 50. Whole training session takes about 10 hours.


    The result of the nearest 20 images in database, respect to the 10 query images are shown as below:


    

<p id="gdcalert22" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert23">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")



    Reverse Video Search

7. <span style="text-decoration:underline;">Conclusion (5%) Summarize your key results - what have you learned? Suggest ideas for future extensions or new applications of your ideas.</span>

    The result of reverse image search was salient, in that we can successfully retrieve the nearest neighbors in the embeddings of images. We have learnt the pipeline of preprocessing datasets, 


    aaa


References



1. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
2. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
3. Going deeper with convolutions

If your project page quotes the requirement for a proposal, the project proposal should be one paragraph (200-400 words). Your project proposal should describe:

What is the problem that you will be investigating? Why is it interesting?

What reading will you examine to provide context and background?

What data will you use? If you are collecting new data, how will you do it?

What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations? You don’t have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on.

How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?

Submission: Please submit your proposal as a Github URL. Only one person on your team should submit. Please have this person add the rest of your team as collaborators in Github as a “Group Submission”.

Your proposal should be (conditionally) approved before you can start working on this project. Incorporate all requested changes to your proposal. You will be receiving the required changes as a Github Pull Request.


## 
    Project Report

Is your paper clearly written and nicely formatted? Writing / Formatting (5%)

Supplementary Material, not counted toward your 6-8 page limit and submitted as a separate file.

Your supplementary material might include:



* Cool videos, interactive visualizations, demos, etc.

Examples of things to not put in your supplementary material:



* The entire PyTorch/TensorFlow Github source code.
* Any code that is larger than 10 MB.
* Model checkpoints.

Submission: You will submit your final report as a markdown + img files under /docs in your Github repo, obviously together with your code and a README file that describes how anyone can run it to replicate your results. It is highly advised to author a Medium article about your work.
