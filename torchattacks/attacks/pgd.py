import torch
import torch.nn as nn
import numpy as np
import scipy.misc
from ..attack import Attack



def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def inverse_transform(images):
    return (images+1.)/2.
    # return ((images + 1.) * 127.5).astype('uint8')

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


class PGD(Attack):
    r"""
    PGD(Linf) attack in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 0.3)
        alpha (float): alpha in the paper. (DEFALUT : 2/255)
        iters (int): step size. (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)
        targeted (bool): using targeted attack with input labels as targeted labels. (DEFAULT : False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGD(model, eps = 4/255, alpha = 8/255, iters=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, opt, eps=4/255, alpha=0.05, iters=20, random_start=False, targeted=False):
        super(PGD, self).__init__("PGD", model, opt)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.random_start = random_start
        self.targeted = targeted
        self.opt = opt
    
    def forward(self, images, labels, batch):
        r"""
        Overridden.
        """
        loss = nn.CrossEntropyLoss()
        if self.targeted :
            loss = lambda x,y : -nn.CrossEntropyLoss()(x,y)
            
        ori_images = images.clone().detach()
                
        if self.random_start:
            # Starting at a uniformly random point
            images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
            images = torch.clamp(images, min=-1, max=1)

        for i in range(self.iters):
            print('[pic:{}]iter{}\n'.format(batch, i))
            images.requires_grad = True
            self.model.set_input([images, labels])
            self.model.forward()
            
            cost = loss(self.model.logits, labels).to(self.device)
            
            grad = torch.autograd.grad(cost, images, 
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images + self.alpha*grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=-1, max=1).detach()
        adv_images = images

        
        return adv_images