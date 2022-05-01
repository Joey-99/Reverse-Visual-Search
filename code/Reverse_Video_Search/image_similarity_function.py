# -*- encoding=utf-8 -*-

# import packages
import cv2
from functools import reduce
from PIL import Image


# calculate similarity function of two images ORB algorithm
def ORB_img_similarity(img1_path,img2_path):
    """
    :param img1_path: image 1 path
    :param img2_path: image 2 path
    :return: image similarity
    """
    try:
        # read image
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # initialize ORB detector
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # extract and calculate feature points
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # knn filter results
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        # view max number of matching points
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        similary = len(good) / len(matches)
        return similary

    except:
        return '0'


# calculate partial hash value of image--pHash
def phash(img):
    """
    :param img: image
    :return: return partial hash value of image
    """
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    return hash_value


# calculate two images similarity function partial sensitive hash algorithm
def phash_img_similarity(img1_path,img2_path):
    """
    :param img1_path: image 1 path
    :param img2_path: image 2 path
    :return: image similarity
    """
    # read image
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # calculate hamming distance
    distance = bin(phash(img1) ^ phash(img2)).count('1')
    similary = 1 - distance / max(len(bin(phash(img1))), len(bin(phash(img2))))
    return similary



# histogram calculate image similarity algorithm
def make_regalur_image(img, size=(256, 256)):
    """It is necessary to unify all the images to a special specification, here I choose a resolution of 256x256."""
    return img.resize(size).convert('RGB')

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def calc_similar(li, ri):
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0

def calc_similar_by_path(lf, rf):
    li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
    return calc_similar(li, ri)

def split_image(img, part_size = (64, 64)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]


