import torchattacks
from data import create_dataset
from options.train_options import TrainOptions
from models import create_model
import numpy as np
import imageio
import copy
import os

def save_images(images, size, image_path, attack='none'):
    return imageio.imsave(image_path, np.uint8(np.squeeze(255 * inverse_transform(images, attack))))


def inverse_transform(images, attack):
    if attack == 'cw':
        return images
    else:
        return (images+1.)/2.


opt = TrainOptions().parse()
opt.batch_size = 1
opt.isTrain = False
train_dataset = create_dataset(opt, flag=opt.flag)
print('Length: {}\n'.format(len(train_dataset)))
model = create_model(opt)
model.setup(opt)
model.eval()


y_class = {}
data_name = opt.dataroot.split('/')[-1]
with open(os.path.join('./datasets/dataset_dict', '{}_dict.txt'.format(data_name)), "r") as f:
    lines = f.readlines()
    for i in lines:
        classes = i.strip().split(":")[0]
        idx = i.strip().split(":")[1]
        y_class[int(idx)] = classes

cla_num = copy.deepcopy(y_class)
for i in cla_num:
    cla_num[i] = 0


if opt.attack == 'fgsm':
    attack = torchattacks.FGSM(model, opt)
elif opt.attack == 'pgd':
    attack = torchattacks.PGD(model, opt)
elif opt.attack == 'cw':
    attack = torchattacks.CW(model, opt)


for i,data in enumerate(train_dataset):
    images, labels = data
    images, labels = images.to('cuda'), labels.to('cuda')
    lab_idx = labels.detach().cpu().numpy()[0]
    clas = y_class[lab_idx]
    if not os.path.exists(os.path.join(opt.dataroot, 'adv_{}_{}'.format(opt.model, opt.attack), 'orig/{}/{}_{}'.format(data_name, opt.flag, opt.train_ratio), '{}'.format(clas))):
        os.makedirs(os.path.join(opt.dataroot, 'adv_{}_{}'.format(opt.model, opt.attack), 'orig/{}/{}_{}'.format(data_name, opt.flag, opt.train_ratio), '{}'.format(clas)))
    save_images(images.detach().cpu().numpy().transpose(0, 2, 3, 1), [1, 1],
                opt.dataroot + '/adv_{}_{}'.format(opt.model, opt.attack) + '/orig/{}/{}_{}/{}/{}.png'.format(data_name, opt.flag, opt.train_ratio, clas, cla_num[lab_idx]))

    adversarial_images = attack(images, labels, i)
    del data

    if not os.path.exists(os.path.join(opt.dataroot, 'adv_{}_{}'.format(opt.model, opt.attack), 'adv/{}/{}_{}'.format(data_name, opt.flag, opt.train_ratio), '{}'.format(clas))):
        os.makedirs(os.path.join(opt.dataroot, 'adv_{}_{}'.format(opt.model, opt.attack), 'adv/{}/{}_{}'.format(data_name, opt.flag, opt.train_ratio), '{}'.format(clas)))
    save_images(adversarial_images.detach().cpu().numpy().transpose(0, 2, 3, 1), [1, 1],
                opt.dataroot + '/adv_{}_{}'.format(opt.model, opt.attack) + '/adv/{}/{}_{}/{}/{}.png'.format(data_name, opt.flag, opt.train_ratio, clas, cla_num[lab_idx]), opt.attack)

    cla_num[lab_idx] += 1