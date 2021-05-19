import os
import torch
import torch.nn as nn
import torchvision
from .base_model import BaseModel
import torch.nn.functional as F



class resnet152model(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.target_model = torchvision.models.resnet152(pretrained=False)
        self.target_model.load_state_dict(torch.load('./pretrained/resnet152-b121ed2d.pth'))
        self.num_fits = self.target_model.fc.in_features
        self.target_model.fc = nn.Linear(self.num_fits, opt.classes)
        self.model_names = ['resnet152']
        self.loss_names = ['loss']
        self.data_name = self.opt.dataroot.split('/')[-1]
        self.y_class = {}
        with open(os.path.join('./datasets/dataset_dict', '{}_dict.txt'.format(self.data_name)), "r") as f:
            lines = f.readlines()
            for i in lines:
                classes = i.strip().split(":")[0]
                idx = i.strip().split(":")[1]
                self.y_class[int(idx)] = classes

        self.acc_names = ['totalacc', 'perclass']
        self.acc_perclass = [0 for i in range(opt.classes)]
        if self.isTrain:
            self.ignored_params = list(map(id, self.target_model.fc.parameters()))
            self.base_params = list(filter(lambda p: id(p) not in self.ignored_params, self.target_model.parameters()))
            self.optimizers = torch.optim.SGD([
                {'params': self.base_params},
                {'params': self.target_model.fc.parameters(), 'lr': opt.lr * 80}], lr=opt.lr, momentum=0.9, weight_decay=1e-4)

        assert (torch.cuda.is_available())
        self.target_model.cuda()
        self.netresnet152 = self.target_model

    def set_input(self, data):
        self.imgs = data[0].to(self.device)
        self.label = data[1].to(self.device)

    def forward(self):
        self.logits = self.netresnet152(self.imgs)

    def backward(self):
        self.loss_loss = F.cross_entropy(self.logits, self.label).to(self.device)
        self.loss_loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizers.zero_grad()
        self.backward()
        self.optimizers.step()

    def test(self, final=True):
        with torch.no_grad():
            self.forward()
            self.pred = torch.argmax(self.logits, 1)
            self.acc_totalacc = torch.sum(self.pred == self.label, 0)
            if final:
                self.acc_perclass = [0 for i in range(self.opt.classes)]
                if self.opt.batch_size == 1:
                    self.acc_perclass[self.label] = float(torch.sum(self.pred == self.label, 0))
                else:
                    for i, index in enumerate(self.label):
                        self.acc_perclass[index.item()] += float(torch.sum(self.pred[i] == self.label[i], 0))
