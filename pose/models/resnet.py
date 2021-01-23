from torch import nn
from torchvision import models
from utils.torch import *
# from torchsummary import summary


class ResNet(nn.Module):

    def __init__(self, out_dim, fix_params=False, running_stats=False, spec='resnet50'):
        super().__init__()
        self.out_dim = out_dim
        if spec == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        else:
            self.resnet = models.resnet18(pretrained=True)
        if fix_params:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)
        self.bn_stats(running_stats)

    def forward(self, x):
        return self.resnet(x)

    def bn_stats(self, track_running_stats):
        for m in self.modules():
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = track_running_stats


if __name__ == '__main__':
    import time
    net = ResNet(128)
    t0 = time.time()
    input = ones(10, 3, 64, 64)
    out = net(input)
    print(time.time() - t0)
    print(out.shape)

    # summary(net, (3, 64, 64))
