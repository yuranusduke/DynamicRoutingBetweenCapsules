"""
Utilities functions

Created by Kunhong Yu
Date: 2021/07/05
"""
import os

import torch as t
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from torch.nn import functional as F
from PIL import Image
import numpy as np


#################
#     Config    #
#################
class Config(object):
    """
    Args :
        --dataset: default is 'mnist' also support 'fashion_mnist'
        --beta: hyper-parameter to trade off negative labels, default is 0.5
        --epochs: training epochs, default is 20
        --batch_size: default is 100
    """
    dataset = 'mnist'

    beta = 0.5
    epochs = 20
    batch_size = 100

    device = 'cuda' if t.cuda.is_available() else 'cpu'

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print(k + ' does not exist, will be added!')

            setattr(self, k, v)


##################
#      Loss      #
##################
def margin_loss(x, y, beta):
    """Define margin loss according to the paper
    Args :
        --x: input tensor, shape [m, 10, 16]
        --y: input label tensor, shape [m, ]
        --beta: hyper-parameter to trade off negative labels
    return :
        --loss
    """
    T = one_hot(y, 10) # [m, 10]
    loss = T * (F.relu(0.9 - x) ** 2) + beta * (1 - T) * (F.relu(x - 0.1) ** 2) # margin loss
    loss = t.mean(t.sum(loss, dim = -1))

    return loss


##################
#    Visualize   #
##################
def show_batch(batch, index = 0, dataset = 'mnist'):
    """Show batch of images
    Args :
        --battch: batch image in numpy form
        --index: index of batch, default is 0
        --dataset: default is 'mnist'
    """
    f, ax = plt.subplots(len(batch), 1, figsize = (5, 20))
    f.suptitle(f'Batch {index + 1} \n Input || Recon', fontsize = 30)
    for i in range(len(batch)):
        x = batch[i].permute(1, 2, 0) if dataset == 'cifar10' else batch[i]
        ax[i].imshow(x)
        ax[i].axis('off')

    if not os.path.exists('./results/re_imgs/'):
        os.makedirs('./results/re_imgs/')
    plt.savefig(f'./results/re_imgs/re_img_{index + 1}_{dataset}.png')
    plt.close()

def show_manipulation(x, model, index = 0, dataset = 'mnist'):
    """This function is used to show manipulation of our capsules in the DigitCaps layer
    Args :
        --x: input DigitCaps, which have shape [1, 10, 16]
        --model: model instance
        --index: batch index
        --dataset: 'mnist' as default
    """
    model.eval()
    step_size = 0.05
    #re_model = model.reconstruction
    f, ax = plt.subplots(16, 11, figsize = (11, 16))
    f.suptitle(f'Recon for all variations in batch {index}')
    with t.no_grad():
        for i in range(16):
            start = -0.3
            for j in range(11):
                start += step_size
                x[0 : 1, :, i] = x[0 : 1, :, i] + start
                x_ = model.reconstruction_module(x)
                x_ = x_.view(3, 28, 28).cpu().numpy() if dataset == 'cifar10' else x_.view(28, 28).cpu().numpy()
                if dataset == 'cifar10':
                    x_ = x_.transpose(1, 2, 0)
                ax[i][j].imshow(x_)
                ax[i][j].grid(False)
                ax[i][j].axis('off')
                x[0: 1, :, i] = x[0: 1, :, i] - start

    plt.savefig(f'./results/re_imgs/re_img_man_{index + 1}_{dataset}.png')
    plt.close()

def generate_animnated(dataset = 'mnist', man = False):
    """
    Generate one animated image
    Args :
        --dataset: default is 'mnist'
        --man: manipulate True or False
    """
    path = os.path.join('./results/re_imgs')
    files = os.listdir(path)
    files = list(filter(lambda x : x.count(dataset) and (x.count('man') if man else not x.count('man')), files))

    plt.ion()

    for file in files[:30]:
        plt.cla()
        img = Image.open(os.path.join(path, file))
        img = np.asarray(img)
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        plt.pause(0.6)

    plt.ioff()
    plt.close()


if __name__ == '__main__':
    generate_animnated(dataset = 'fashion_mnist', man = False)