# Report

## CIFAR100

**For CIFAR100**, the best configuration of parameters from the ones tested is:

- **model:** ResNet18;
- **optimizer:** SGDNesterov;
- **scheduler:** ReduceLROnPlateau;
- **batch size:** 64;
- **learning rate:** 0.001;
- **patience (for early stopping):** 5.

It got accuracy 0.5369. Unfortunately, all tests used only 10 epochs, due to lack of time.

| Id | Model          | Optimizer   | Scheduler         | Epochs | Accuracy |
|----|----------------|-------------|-------------------|--------|----------|
| 1  | ResNet18       | SGDNesterov | ReduceLROnPlateau | 10     | 0.5369   |
| 2  | ResNet18       | RMSProp     | StepLR            | 10     | 0.3929   |
| 3  | PreActResNet18 | AdamW       | StepLR            | 10     | 0.5110   |
| 4  | PreActResNet18 | SGDMomentum | ReduceLROnPlateau | 10     | 0.4626   |

Configurations 1 and 4 got the most consistent accuracy growing rates.

## MNIST

**For MNIST**, it was way easier, and the following configuration already led to an accuracy of 0.9102:

- **model:** ResNet18;
- **optimizer:** SGDNesterov;
- **scheduler:** ReduceLROnPlateau;
- **batch size:** 64;
- **learning rate:** 0.001;
- **patience (for early stopping):** 5.
