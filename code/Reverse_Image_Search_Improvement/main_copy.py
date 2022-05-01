# %%
import torch.utils.data as Data
import torch
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.datasets import LFWPeople, LFWPairs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from scipy import ndimage
import numpy as np
from PIL.ImageEnhance import *
from torchvision.transforms import *
import random
from skimage import transform as trans
from mtcnn_pytorch.src import detect_faces, show_bboxes
from PIL import Image
import cv2
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

from models.inception_resnet_v1 import InceptionResnetV1

manualSeed=0
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
np.random.RandomState(manualSeed)

# %%
## creating dataset
lfw_people = LFWPairs(root='/home/weicheng/selfLearning/facenet/project/lfw', 
                       download=True, transform=T.Compose([
                T.Resize(size=(250,250)),
                T.ToTensor(),
            ]))

# %%
## create model, we can use pretrained or not
resnet = InceptionResnetV1(classify=False, pretrained='vggface2')
resnet = resnet.cuda(0)
resnet.train()

# %%
## create dataloader to minimize the use of cpu
loader = Data.DataLoader(
    dataset=lfw_people,
    batch_size=32,
    shuffle=True, drop_last=True)

## optimizer and loss function
optimizer = optim.Adam(resnet.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-09, weight_decay=1e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 50, 100, 200, 500])
criterion = torch.nn.MSELoss()


# %%
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * torch.nn.functional.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
contra = ContrastiveLoss(margin=10)

## use MTCNN to find aligned faces
def process_ndarray(img1_cv2):
    for i in range(img1_cv2.shape[0]):
        # print(img1_cv2[i].squeeze().shape) # (160, 160, 3)
        img = img1_cv2[i].squeeze() * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        # print(img.size)
        src = np.array([                                                                                                                                                                     
            [30.2946, 51.6963],                                                                                                                                                                
            [65.5318, 51.5014],                                                                                                                                                                
            [48.0252, 71.7366],                                                                                                                                                                
            [33.5493, 92.3655],                                                                                                                                                                
            [62.7299, 92.2041] ], dtype=np.float32 )
        src[:,0] *= (img.size[0]/96)
        src[:,1] *= (img.size[1]/112)
        _, landmarks = detect_faces(img)
        # print(landmarks)
        dst = landmarks[0].astype(np.float32)
        facial5points = [[dst[j],dst[j+5]] for j in range(5)]
        tform = trans.SimilarityTransform()                                                                                                                     
        tform.estimate(np.array(facial5points), src)
        M = tform.params[0:2,:]
        warped = cv2.warpAffine(img1_cv2[i].squeeze(),M,(img.size[0],img.size[1]), borderValue = 0.0)
        # print(warped.shape)
        # print(np.max(warped))
        img1_cv2[i] = warped
    return img1_cv2

    
# %%
## training session
for epoch in range(200):
    list_loss = []
    for step, (batch_x, batch_y, target) in enumerate(tqdm(loader)):
        img1, img2, target = batch_x, batch_y, target.cuda()
        img1 = img1.view(img1.shape[0], img1.shape[2], img1.shape[3], img1.shape[1])
        img2 = img2.view(img2.shape[0], img2.shape[2], img2.shape[3], img2.shape[1])
        img1_cv2 = img1.numpy()
        img2_cv2 = img2.numpy()
        img1_cv2 = process_ndarray(img1_cv2)
        img2_cv2 = process_ndarray(img2_cv2)
        img1 = torch.FloatTensor(img1_cv2).cuda()
        img1 = img1.view(img1.shape[0], img1.shape[3], img1.shape[1], img1.shape[2])
        img2 = torch.FloatTensor(img2_cv2).cuda()
        img2 = img2.view(img2.shape[0], img2.shape[3], img2.shape[1], img2.shape[2])
        pred1, output1 = resnet(img1)
        pred2, output2 = resnet(img2)
        consist1 = criterion(output1, img1)
        consist2 = criterion(output2, img2)
        # pred_softmax = torch.softmax(pred, dim=-1)
        loss = contra(pred1, pred2, target) + (consist1+consist2)
        list_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    # break
    print('loss in epoch {} is '.format(epoch), sum(list_loss)/len(list_loss))
    if(epoch%10==0):
        save_mode_path = os.path.join('../model_mtcnn', 'epoch_' + str(epoch) + '.pth')
        torch.save(resnet.state_dict(), save_mode_path)

