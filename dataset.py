import os.path
import cv2
import glob
import random
from torch.utils.data import Dataset
import torch

class AugGANDataset(Dataset):
    def __init__(self, dataset_name=None, transform=None, mode='train', crop_size=(600,600), resize_size=(256,128)):

        self.transform = transform
        self.mode = mode
        self.crop_size = crop_size
        self.resize_size = resize_size

        dataset_dir = os.path.join('dataset', dataset_name)

        self.imgs_A = sorted(glob.glob(os.path.join(dataset_dir, 'trainA', f"imagesA/*.*")))
        self.labels_A = sorted(glob.glob(os.path.join(dataset_dir, 'trainA', f"labelsA/*.*")))
        self.imgs_B = sorted(glob.glob(os.path.join(dataset_dir, 'trainB', f"imagesB/*.*")))
        #self.labels_B = sorted(glob.glob(os.path.join(dataset_dir, 'trainB', f"labelsB/*.*")))

    def __getitem__(self, index):

        # load images and labels
        item_A = cv2.imread(self.imgs_A[index])
        label_A = cv2.imread(self.labels_A[index])

        if self.mode == 'train':
            idx_B = random.randint(0, len(self.imgs_B)-1)
            item_B = cv2.imread(self.imgs_B[idx_B])
            #label_B = cv2.imread(self.labels_B[idx_B])
        elif self.mode == 'val':
            idx_B = random.randint(0, len(self.imgs_B)-1)
            item_B = cv2.imread(self.imgs_B[idx_B])
        else: #self.mode = 'test'
            item_B = cv2.imread(self.imgs_B[index])

        # BGR -> RGB
        item_A = cv2.cvtColor(item_A, cv2.COLOR_BGR2RGB)
        item_B = cv2.cvtColor(item_B, cv2.COLOR_BGR2RGB)

        # crop
        '''
        h, w, _ = item_A.shape

        if self.mode == 'train' or self.mode == 'val':
            new_h, new_w = self.crop_size
            top   = random.randint(0, h - new_h)
            left  = random.randint(0, w - new_w)
            item_A  = item_A[top:top + new_h, left:left + new_w]
            item_B  = item_B[top:top + new_h, left:left + new_w]
            label_A = label_A[top:top + new_h, left:left + new_w]
            label_B = label_B[top:top + new_h, left:left + new_w]
        else :
            new_h, new_w = self.crop_size / 2
            test_w_left, test_w_right = int(w/2-new_h), int(w/2+new_h)
            test_h_left, test_h_right = int(h/2-new_w), int(h/2+new_w)
            item_A   = item_A[test_h_left:test_h_right, test_w_left:test_w_right]
            item_B   = item_B[test_h_left:test_h_right, test_w_left:test_w_right]
        '''

        # resize
        item_A  = cv2.resize(item_A, self.resize_size, cv2.INTER_LINEAR)
        item_B  = cv2.resize(item_B, self.resize_size, cv2.INTER_LINEAR)
        label_A = cv2.resize(label_A, self.resize_size, cv2.INTER_NEAREST)
        #label_B = cv2.resize(label_A, self.resize_size, cv2.INTER_NEAREST)


        # transform
        item_A = self.transform(item_A)
        item_B = self.transform(item_B)

        label_A = cv2.cvtColor(label_A, cv2.COLOR_BGR2GRAY)
        #label_B = cv2.cvtColor(label_B, cv2.COLOR_BGR2GRAY)
        label_A = torch.from_numpy(label_A.copy()).long()
        #label_B = torch.from_numpy(label_B.copy()).long()

        # prepare the targets (for segmentation)
        h, w = label_A.size()

        target_A = torch.zeros(20, h, w) #20 is segment label numbers (BDD dataset has 20 labels)
        #target_B = torch.zeros(20, h, w)
        for c in range(20):
            target_A[c][label_A == c] = 1
            #target_B[c][label_B == c] = 1

        #return {'A': item_A, 'B': item_B, 'lA': target_A, 'lB': target_B}
        return {'A': item_A, 'B': item_B, 'lA': target_A}

    def __len__(self):
        return max(len(self.imgs_A), len(self.imgs_B))
