# -*- encoding=utf-8 -*-

from image_similarity_function import *
import os
import shutil

# fusion similarity threshold
threshold1 = 0.85
# final similarity higher to determine the threshold
threshold2 = 0.98


# fusion func to calculate image similarity
def calc_image_similarity(img1_path, img2_path):
    """
    :param img1_path: filepath+filename
    :param img2_path: filepath+filename
    :return: image final similarity
    """

    similary_ORB = float(ORB_img_similarity(img1_path, img2_path))
    similary_phash = float(phash_img_similarity(img1_path, img2_path))
    similary_hist = float(calc_similar_by_path(img1_path, img2_path))
    # in 3 algorithms, if max similarity >= 0.85,take similarity as max; otherwise, take the min
    max_three_similarity = max(similary_ORB, similary_phash, similary_hist)
    min_three_similarity = min(similary_ORB, similary_phash, similary_hist)
    if max_three_similarity > threshold1:
        result = max_three_similarity
    else:
        result = min_three_similarity

    return round(result, 3)


if __name__ == '__main__':

    # search image path and file name
    img1_path = './video2pic/testImage1.jpg'

    # search folder
    filepath = './video2pic/'

    # similar image storage path
    newfilepath = './similardata/'

    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            # print(filepath+filename)
            img2_path = filepath + filename
            kk = calc_image_similarity(img1_path, img2_path)

            try:
                if kk >= threshold2:
                    print(img2_path, kk)
                    shutil.copy(img2_path, newfilepath)
            except Exception as e:
                # print(e)
                pass
