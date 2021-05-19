import torch
from .base_model import BaseModel
from . import networks
from . import target_model
import torch.nn as nn
import torch.nn.functional as F


class PSGANModel(BaseModel):
    """ 
    This class implements the PSGAN model.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values to match the PSGAN paper
        parser.set_defaults(norm='batch')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.alpha = 0.9
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D_adv', 'G_L1', 'G_Dfeat', 'C_real', 'C_adv', 'C_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'C']
        else:  # during test time, only load C
            self.model_names = ['C']

        # define networks (both generator, discriminator, and target model)
        net = getattr(target_model, '{}'.format(self.opt.target_model))(pretrained=False)
        net.load_state_dict(torch.load('./pretrained/{}.pth'.format(self.opt.target_model)))
        num_fits = net.fc.in_features
        net.fc = nn.Linear(num_fits, self.opt.classes)
        self.netC = net.cuda()
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.classes, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.Dcla = nn.CrossEntropyLoss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 4, betas=(opt.beta1, 0.999))
            self.ignored_params = list(map(id, self.netC.fc.parameters()))  # 返回的是parameters的 内存地址
            self.base_params = list(filter(lambda p: id(p) not in self.ignored_params, self.netC.parameters()))
            self.optimizer_C = torch.optim.SGD([
                {'params': self.base_params},
                {'params': self.netC.fc.parameters(), 'lr': opt.lr * 20}], lr=opt.lr, momentum=0.9, weight_decay=1e-4)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_C)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.label = input['y'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not self.isTrain:
            adv_logits = self.netC(self.real_A)
            adv_label = torch.argmax(adv_logits, 1)
            self.acc_adv = torch.sum(adv_label == self.label, 0)
        else:
            self.one_hot_true = torch.zeros([self.label.shape[0], self.opt.classes]).to(self.device)
            for i in range(self.real_B.shape[0]):
                self.one_hot_true[i, self.label[i]] = 1
            self.noise = self.alpha * self.netG(self.real_A, self.one_hot_true)  # G(A)
            self.fake_B = torch.clamp((self.real_A - self.noise), -1, 1)
            self.r_noise = torch.clamp((self.real_A - self.real_B), -1, 1)
            
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        adv_AB = torch.cat((self.real_A, self.real_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        adv_fake, _ = self.netD(adv_AB)
        self.loss_D_adv = self.criterionGAN(adv_fake, False)

        real_AB = torch.cat((self.real_B, self.real_B), 1)
        pred_real, self.real_feat = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        fake_AB = torch.cat((self.fake_B.detach(), self.real_B), 1)
        pred_fake, _ = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_adv + self.loss_D_real + self.loss_D_fake) * 1
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.fake_B, self.real_B), 1)
        pred_fake, fake_feat = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, || noise - r_noise ||_1
        self.loss_G_L1 = self.criterionL1(self.noise, self.r_noise)

        # Third, ||real_Dfeat - fake_Dfeat ||_1
        self.loss_G_Dfeat = nn.L1Loss()(self.real_feat, fake_feat)

        # combine loss and calculate gradients
        self.loss_G = 1 * self.loss_G_GAN + self.opt.l1_lambda * self.loss_G_L1 + self.opt.Dfeat_lambda * self.loss_G_Dfeat

    def backward_C(self):
        adv_logits = self.netC(self.real_A)
        self.loss_C_adv = F.cross_entropy(adv_logits, self.label)

        noise = self.alpha * self.netG(self.real_A, self.one_hot_true)
        fake_B = torch.clamp((self.real_A - noise), -1, 1)
        fake_logits = self.netC(fake_B)
        self.loss_C_fake = F.cross_entropy(fake_logits, self.label)

        real_logits = self.netC(self.real_B)
        self.loss_C_real = F.cross_entropy(real_logits, self.label)

        # combine loss and calculate gradients
        self.loss_C = self.loss_C_adv + self.loss_C_fake + self.loss_C_real
        self.loss_C.backward()

    def optimize_parameters(self, epoch):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        # update C
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing C
        self.optimizer_C.zero_grad()        # set C's gradients to zero
        self.backward_C()                   # calculate graidents for C
        self.optimizer_C.step()             # udpate C's weights