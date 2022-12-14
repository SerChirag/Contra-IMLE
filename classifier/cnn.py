import torchvision.datasets as torchdata


def main():
    trainset = torchdata.CIFAR10(root='./data', train=True, download=True, transform=transform)


main()
