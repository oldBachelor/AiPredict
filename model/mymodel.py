'''
network models.
'''
import torch
from torch import nn
class Mymodel(nn.Module): #400-100
    def __init__(self):
        super(Mymodel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(5,5),stride=1) #(-4)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),stride=3) # (/3)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=False)
    def forward(self,x):
        for i in range(5):#480
            x = self.conv1(x)
        x = self.conv2(x)#160
        return x

class Mymodel2(nn.Module):#224-60
    def __init__(self):
        super(Mymodel2, self).__init__()
        # the vgg's layers
        # self.features = features
        # 13 conv + 3 FC
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        batch_norm = False
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.Batchnorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        # use the vgg layers to get the feature
        self.features = nn.Sequential(*layers)
        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 决策层：分类层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 60),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x_fea = x
        x = self.avgpool(x)
        x_avg = x
        # batch*pixel
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, x_fea, x_avg


class Mymodel3(nn.Module): #270-1,img---value
    def __init__(self):
        super(Mymodel3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64 ,kernel_size=(5,5),stride=1) #(-4)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=3) # (/3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=3)  # (/3)
        self.relu = nn.ReLU(inplace=True)
        self.Linear1 = nn.Linear(64 * 10*10, 50)
        self.Linear2 = nn.Linear(50, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=False)

    def forward(self,x):#270-1,series--value
        x = self.conv2(x)#90*90
        x = self.relu(x)
        x = self.conv3(x)#30*30
        x = self.relu(x)
        x = self.conv3(x)#10*10
        x = self.relu(x)
        x = x.view(-1)
        x = self.Linear1(x) #50
        x = self.relu(x)
        x = self.Linear2(x)#1
        return x

class Mymodel4(nn.Module): #270-1,seq---value
    def __init__(self):
        super(Mymodel4, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3,stride=3)#3
        self.conv2= nn.Conv1d(64, 64, kernel_size=3, stride=3)  # 3
        self.relu = nn.ReLU(inplace=True)
        self.Linear1 = nn.Linear(64 * 10, 50)
        self.Linear2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x) #90
        x = self.relu(x)
        x = self.conv2(x)  # 30
        x = self.relu(x)
        x = self.conv2(x)  # 10
        x = self.relu(x)
        x = x.view(-1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x





if __name__ == "__main__":
    mymodel = Mymodel4()
    input = torch.ones((1,1,270))
    output = mymodel(input)
    print(output)
    print(output.shape)
    print(len(output))

