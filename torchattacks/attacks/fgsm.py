import torch
import torch.nn as nn

from ..attack import Attack

class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (DEFALUT : 0.007)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.05)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, opt, eps=0.02):
        super(FGSM, self).__init__("FGSM", model, opt)
        self.eps = eps
    
    def forward(self, images, labels, batch):
        r"""
        Overridden.
        """
        print('[pic:{}]iter{}\n'.format(batch, 1))
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        
        images.requires_grad = True
        self.model.set_input([images, labels])
        self.model.forward()
        outputs = self.model.logits
        cost = loss(outputs, labels).to(self.device)
        
        grad = torch.autograd.grad(cost, images, 
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=-1, max=1).detach()

        return adv_images