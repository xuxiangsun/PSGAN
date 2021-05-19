import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np


def devide_dataset(path, classes, ratio):
    # check if you have devided the dataset, here we only check training set, but you should make sure
    # that testing set is not exist
    if not os.path.exists(os.path.join(path, 'train_{}'.format(ratio))):
        if not os.path.exists(os.path.join(path, 'train_{}'.format(ratio))):
            os.makedirs(os.path.join(path, 'train_{}'.format(ratio)))
        if not os.path.exists(os.path.join(path, 'test_{}'.format(ratio))):
            os.makedirs(os.path.join(path, 'test_{}'.format(ratio)))
        ori_path = os.path.join(path, 'Images')

        for label in os.listdir(ori_path):
            image_dir = os.path.join(ori_path, label)
            if not os.path.exists(os.path.join(path, 'train_{}'.format(ratio), label)):
                os.makedirs(os.path.join(path, 'train_{}'.format(ratio), label))
            if not os.path.exists(os.path.join(path, 'test_{}'.format(ratio), label)):
                os.makedirs(os.path.join(path, 'test_{}'.format(ratio), label))
            num = 0
            for ima in os.listdir(image_dir):
                file_number = len(os.listdir(image_dir))
                if num < int(file_number * ratio):
                    im = Image.open(os.path.join(image_dir, ima))
                    im.save(os.path.join(path, 'train_{}'.format(ratio), label, '{}.png'.format(num)))
                    num += 1
                else:
                    im = Image.open(os.path.join(image_dir, ima))
                    im.save(os.path.join(path, 'test_{}'.format(ratio), label, '{}.png'.format(num)))
                    num += 1


class AIDDataset(BaseDataset):

    def __init__(self, opt, flag='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.flag = flag
        self.opt = opt
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
        # if you have devided the dataset, then you can shield the following command
        # devide_dataset(self.datasets_dir, self.y_class, ratio=opt.train_ratio)
        if self.flag == 'train':
            self.path = os.path.join(self.datasets_dir, 'train_{}'.format(opt.train_ratio))
        elif self.flag == 'test':
            self.path = os.path.join(self.datasets_dir, 'test_{}'.format(opt.train_ratio))
        self.data = []
        self.label = []
        for label in os.listdir(self.path):
            id = self.y_class[label]
            image_dir = os.path.join(self.path, label)
            for ima in os.listdir(image_dir):
                im = os.path.join(image_dir, ima)
                self.data.append(im)
                self.label.append(id)
        self.label = np.asarray(self.label)

        self.transform = get_transform(self.opt, grayscale=(self.input_nc == 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, label = self.data[index], self.label[index]
        imgs = Image.open(img)
        imgs_resize = imgs.resize((self.opt.load_size, self.opt.load_size))
        imgs = self.transform(imgs_resize)

        return imgs, label

    def __len__(self):
        return len(self.data)
