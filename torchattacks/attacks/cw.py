import torch
import torch.nn as nn
import warnings

import torch.optim as optim

from ..attack import Attack

class CW(Attack):
    r"""
    CW(L2) attack in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

        
    Arguments:
        model (nn.Module): model to attack.
        targeted (bool): True - change image closer to a given label. False  - change image away from a given label. (DEFALUT : False)
        c (float): c in the paper. parameter for box-constraint. (DEFALUT : 1e-4)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`    
        kappa (float): kappa (also written as 'confidence') in the paper. (DEFALUT : 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        iters (int): max iterations. (DEFALUT : 1000)
        lr (float): learning rate of the Adam optimizer. (DEFALUT : 0.01)
        
    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.CW(model, targeted=False, c=10, kappa=0, iters=500, lr=0.02)
        >>> adv_images = attack(images, labels)
        
    .. note:: There are serveral NOT IMPLEMENTED part of the paper/other codes as follows :
    
        (1) Binary search method for c : It costs too many times.
        
        (2) Optimization on tanh space : Not in the paper, but in the other codes.
        
        (3) Choosing method best l2 adversaries : It costs too many times.
    """
    def __init__(self, model, opt, targeted=False, c=0.1, kappa=20, iters=20, lr=0.1):
        super(CW, self).__init__("CW", model, opt)
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.iters = iters
        self.lr = lr
        self.opt = opt
                
    def forward(self, images, labels, batch):
        r"""
        Overridden.
        """
        images = ((images + 1.)/2.).cuda()
        labels_ = labels.cuda()

        # f-function in the paper
        def f(x, labels):
            self.model.set_input([2 * x - 1., labels])
            self.model.forward()
            outputs = self.model.logits
            one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()
            c = one_hot_labels.bool()
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())

            # If targeted, optimize for making the other class most likely 
            if self.targeted:
                return torch.clamp(i-j, min=-self.kappa)

            # If untargeted, optimize for making the other class most likely 
            else :
                return torch.clamp(j-i, min=-self.kappa)
        
        w = torch.zeros_like(images).cuda()
        w.detach_()
        w.requires_grad=True
        
        optimizer = optim.Adam([w], lr=self.lr)
        prev = 1e10
                
        for step in range(self.iters):
            print('pic:{},iter:{}\t'.format(batch, step))
            a = (nn.Tanh()(w) + 1.)/2.

            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(self.c*f(a, labels_))
            
            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (self.iters//10) == 0 :
                if cost > prev :
                    warnings.warn("\n * Early Stopped cause loss does not converge", Warning)
                    return ((nn.Tanh()(w) + 1.)/2.).detach()
                prev = cost
            
            print('- CW Attack Progress : %2.2f %%\n' %((step+1)/self.iters*100))
            
        adv_images = ((nn.Tanh()(w) + 1.)/2.).detach()
        return adv_images