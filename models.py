# -------------------------------
# WideResNet
# -------------------------------

# на основе https://github.com/bmsookim/wide-resnet.pytorch/

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, dropout_rate):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            layers.append(block(in_planes if i == 0 else out_planes, out_planes, s, dropout_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        # проверим корректность глубины сети (должна быть 6n+4)
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6 # колво блоков в каждой группе
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k] # колво каналов на каждом этапе

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, dropout_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, dropout_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, dropout_rate)

        # Финальные слои
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out) #тут уменьшили размерность
        out = self.block3(out) #тут уменьшили размерность
        out = self.bn1(out)
        out = self.relu(out)
        out = F.adaptive_avg_pool2d(out, 1) # глобальное усреднение
        out = out.view(out.size(0), -1)   # развернули в вектор
        return self.fc(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) # задаем веса batchnorm как 1
                nn.init.constant_(m.bias, 0)   # смещения bn как 0
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
