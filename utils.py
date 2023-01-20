import torch

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def num_params(model):
        params = 0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            params += nn
        return params

class LambdaLR():
    def __init__(self, n_epochs, decay_start_epoch=None,offset=0):
        
        self.n_epochs = n_epochs
        self.offset = offset
        if decay_start_epoch is None:
            self.decay_start_epoch=int(n_epochs/2)
        else:
            self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)