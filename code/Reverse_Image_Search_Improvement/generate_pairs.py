import os
import random
import argparse  
import sys
class GeneratePairs:
    """
    Generate the pairs.txt file for applying "validate on LFW" on your own datasets.
    """
    #写成命令行格式就用args解析参数
    # def __init__(self, args):
    #     """
    #     Parameter data_dir, is your data directory.
    #     Parameter pairs_filepath, where is the pairs.txt that belongs to.
    #     Parameter img_ext, is the image data extension for all of your image data.
    #     """
    #     self.data_dir = args.data_dir
    #     self.data_dir =self.data_dir + "/"
    #     self.pairs_filepath = args.saved_dir + "/" + 'pairs.txt'
    #     self.repeat_times = int(args.repeat_times)
    #     self.img_ext = '.png'
    #在pycharm上直接运行，就用这种直接修改参数的方法比较方便，自己选择
    def __init__(self):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = '/home/weicheng/selfLearning/facenet/project/lfw/lfw'
        self.data_dir =self.data_dir + "/"    #验证集路径
        self.pairs_filepath = 'my_pairs.txt'    #pairs.txt存放路径
        self.repeat_times = int(10)
        # self.img_ext = '.png'  #因为我自己验证集png和jpg格式都有，所以不固定图片格式后缀
    def generate(self):
        # The repeate times. You can edit this number by yourself
        folder_number = self.get_folder_numbers()
        print('folder_number--',folder_number)
        # This step will generate the hearder for pair_list.txt, which contains
        # the number of classes and the repeate times of generate the pair
        #如果存在旧的pairs先删除
        if os.path.exists(self.pairs_filepath):    
            os.remove(self.pairs_filepath)
        #删完重开一个pair.txt
        if not os.path.exists(self.pairs_filepath):
            with open(self.pairs_filepath,"a") as f:
                f.write(str(self.repeat_times) + "\t" + str(folder_number) + "\n")
        for i in range(self.repeat_times):
            print('第 %d 次：'%int(i))
            self._generate_matches_pairs()
            self._generate_mismatches_pairs()

    def get_folder_numbers(self):
        count = 0
        for folder in os.listdir(self.data_dir):
            if os.path.isdir(self.data_dir + folder):
                count += 1
        return count
    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue
            a = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)
            with open(self.pairs_filepath, "a") as f:
                temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                w = temp[0]
                # l = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                # r = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                l = random.choice(a).split("_")[-1].lstrip("0").split(".")[0]
                r = random.choice(a).split("_")[-1].lstrip("0").split(".")[0]
                print ('写入 %s ,%s ,%s'%(w,l,r))
                f.write(w + "\t" + l + "\t" + r + "\n")

    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs

        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store" or name[-3:] == 'txt':
                continue
            remaining = os.listdir(self.data_dir)
            del remaining[i]
            remaining_remove_txt = remaining[:]
            for item in remaining:
                if item[-3:] == 'txt':
                    remaining_remove_txt.remove(item)
            remaining = remaining_remove_txt
            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                file1 = random.choice(os.listdir(self.data_dir + name))
                name1=file1.split("_")[0]
                file2 = random.choice(os.listdir(self.data_dir + other_dir))
                name2=file2.split("_")[0]
                f.write(name1 + "\t" + file1.split("_")[-1].lstrip("0").split(".")[0] \
                        + "\t" + name2 + "\t" + file2.split("_")[-1].lstrip("0").split(".")[0] + "\n")


if __name__ == '__main__':
    gen=GeneratePairs()
    gen.generate()