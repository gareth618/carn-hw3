import torch
import yaml
from torch import nn, optim, utils
from torchvision import datasets, models, transforms

class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def make_resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(PreActResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.layers = self._make_layer(block, 64, num_blocks[0])
        self.layers += self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layers += self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layers += self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

with open('config.yaml', 'r') as fd:
    config = yaml.safe_load(fd)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = {
        'MNIST': datasets.MNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
    }[config['dataset']]

    model = {
        'MLP': MLP(),
        'LeNet': LeNet(),
        'ResNet18': make_resnet18(),
        'PreActResNet18': PreActResNet(PreActBlock, [2, 2, 2, 2]),
    }[config['model']].to(device)

    optimizer = {
        'Adam': optim.Adam(model.parameters(), lr=config['learning_rate']),
        'SGD': optim.SGD(model.parameters(), lr=config['learning_rate']),
        'SGDMomentum': optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9),
        'SGDNesterov': optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, nesterov=True),
        'AdamW': optim.AdamW(model.parameters(), lr=config['learning_rate']),
        'RMSProp': optim.RMSprop(model.parameters(), lr=config['learning_rate']),
    }[config['optimizer']]

    scheduler = {
        'None': None,
        'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
    }[config['scheduler']]

    train_dataset = dataset(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    valid_dataset = dataset(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)

    counter = 0
    min_loss = 1e9

    for epoch in range(config['epoch_count']):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('epoch:', epoch, 'train_loss:', train_loss / len(train_loader))

        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                valid_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        accuracy = correct / len(valid_loader) / config['batch_size']
        avg_valid_loss = valid_loss / len(valid_loader)
        print('epoch:', epoch, 'valid_loss:', avg_valid_loss, 'accuracy:', accuracy)
        if scheduler is not None:
            scheduler.step(avg_valid_loss)

        if avg_valid_loss > min_loss:
            counter += 1
            if counter >= config['patience']:
                break
        else:
            min_loss = avg_valid_loss
            counter = 0
