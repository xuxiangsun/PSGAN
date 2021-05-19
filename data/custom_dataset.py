import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np

class CustomDataset(BaseDataset):

    def __init__(self, opt, flag='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.flag = flag
        self.datasets_dir = opt.dataroot
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.y_class = {}
        self.data_name = self.datasets_dir.split('/')[-1]
        with open(os.path.join('./datasets/dataset_dict', '{}_dict.txt'.format(self.data_name)), "r") as f:
            lines = f.readlines()
            for i in lines:
                classes = i.strip().split(":")[0]
                idx = i.strip().split(":")[1]
                self.y_class[classes] = int(idx)

        self.pathA = os.path.join(self.datasets_dir, 'adv_{}_{}'.format(opt.target_model, opt.attack), 'adv', self.data_name, '{}_{}'.format(self.flag, opt.train_ratio))
        self.pathB = os.path.join(self.datasets_dir, 'adv_{}_{}'.format(opt.target_model, opt.attack), 'orig', self.data_name, '{}_{}'.format(self.flag, opt.train_ratio))
        self.dataA = []
        self.dataB = []
        self.label = []
        for label in os.listdir(self.pathA):
            id = self.y_class[label]
            image_dirA = os.path.join(self.pathA, label)
            image_dirB = os.path.join(self.pathB, label)
            for ima in os.listdir(image_dirA):
  
                imA = os.path.join(image_dirA, ima)
                imB = os.path.join(image_dirB, ima)
                self.dataA.append(imA)
                self.dataB.append(imB)
                self.label.append(id)

        self.label = np.asarray(self.label)
        assert len(self.dataA) == len(self.dataB)

        self.transform = get_transform(self.opt, grayscale=(self.input_nc == 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (adv_image, orig_image, target) where target is label of the orig_image.
        """
        imgA, imgB, label = self.dataA[index], self.dataB[index], self.label[index]
        imA = Image.open(imgA)
        imA_resize = imA.resize((self.opt.load_size, self.opt.load_size))
        imB = Image.open(imgB)
        imB_resize = imB.resize((self.opt.load_size, self.opt.load_size))
        imgA = self.transform(imA_resize)
        imgB = self.transform(imB_resize)
        return {'A': imgA, 'B': imgB, 'y': label}


    def __len__(self):
        return len(self.dataA)
