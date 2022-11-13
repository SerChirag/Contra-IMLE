
Weights: https://1sfu-my.sharepoint.com/:u:/g/personal/cva19_sfu_ca/EUqqwCBqdxFHia2Xm1UXLugBE9ABm0roXE4ToGHFaDS3kw?e=BDehHH



For Classifier

#Download pretrained weights
python train.py --download_weights 1 

#Test pretrained models
python train.py --test_phase 1 --pretrained 1 --classifier vgg13_bn

#Train model
python train.py --classifier vgg13_bn 

