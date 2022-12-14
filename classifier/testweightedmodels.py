from vgg import vgg13_bn

my_model = vgg13_bn(pretrained=True)
my_model.eval()

