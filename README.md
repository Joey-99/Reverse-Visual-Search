# Reverse-Visual-Search
*CSGY 6613 AI project Team 8*
## 1. Team members:
* Name: Weicheng Dai  
Email: wd2119@nyu.edu
* Name: Hansheng Li  
Email: hl4346@nyu.edu
* Name: Yunyi Zhu  
Email: yz7736@nyu.edu
## 2. AWS contact person
* Name: Hansheng Li  
Email: hl4346@nyu.edu  
## 3. Project content
* ### code folder  
It includes our code which can be divied into 3 parts: Reverse Image Search - Baseline, Reverse Image Search Improvement and Reverse Video Search for the extra credit.
* ### docs folder  
It contains final report, supplementary material and img files.
## 4. How to run and replicate results  
* ### Reverse Image Search - Baseline  
File name is "Reverse Image Search Baseline.ipynb" in code file, since it is a ipynb file, so you can just dowload the file and import to colab to run. In the github you can preview the code process and output.

* ### Reverse Image Search Improvement   
To run this improved version, you should first run
```
python main_copy.py
```
After finishing the training session, you can use the model to run testing file. Follow the instructions in [test.ipynb](https://github.com/Joey-99/Reverse-Visual-Search/blob/master/weicheng/code/test.ipynb) (please download this file as it is large), choose the query images and number of returns as you like.

* ### Reverse Video Search  
To reproduce this part, you should cut video to images per frame and store in `video2pic` folder. Make sure the required packages are installed like [cv2](https://pypi.org/project/opencv-python/), [PIL](https://pillow.readthedocs.io/en/stable/installation.html). Then run
```
python image_similarity_main.py
```

