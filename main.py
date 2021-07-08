"""
Main file

Created by Kunhong Yu
Date: 2021/07/01
"""
import torch as t
import torchvision as tv
from utils import Config, show_batch, margin_loss, show_manipulation
from capsules import CapsNet
import tqdm
import matplotlib.pyplot as plt
import sys
import fire
from datetime import datetime


# Step 0 Decide the structure of the model#
opt = Config()
device = t.device(opt.device)

# Step 1 Load the data set#
def get_data_loader(dataset : str, train : bool, batch_size : int) -> t.utils.data.DataLoader:
    """Get data loader
    Args :
        --dataset: string format
        --train: True for train, False for test
        --batch_size: training or testing batch size
    return :
        --data_loader
    """
    if train:
        if dataset == 'mnist':
            train_data = tv.datasets.MNIST(root = './data/',
                                           train = True,
                                           download = True,
                                           transform = tv.transforms.ToTensor())
        elif dataset == 'fashion_mnist':
            train_data = tv.datasets.FashionMNIST(root = './data/',
                                                  train = True,
                                                  download = True,
                                                  transform = tv.transforms.ToTensor())
        elif dataset == 'cifar10':
            train_data = tv.datasets.CIFAR10(root = './data/',
                                             train = True,
                                             download = True,
                                             transform = tv.transforms.Compose([
                                                 tv.transforms.RandomCrop(28, padding = 4),
                                                 #tv.transforms.RandomCrop(28, padding = 4),
                                                 tv.transforms.RandomHorizontalFlip(),
                                                 tv.transforms.RandomRotation(15),
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(0.5, 0.5)
                                             ]))

        else:
            raise Exception('No other data sets!')
        data_loader = t.utils.data.DataLoader(train_data,
                                              batch_size = batch_size,
                                              shuffle = True,
                                              drop_last = False)
    else:
        if dataset == 'mnist':
            test_data = tv.datasets.MNIST(root = './data/',
                                          train = False,
                                          download = True,
                                          transform = tv.transforms.ToTensor())
        elif dataset == 'fashion_mnist':
            test_data = tv.datasets.FashionMNIST(root = './data/',
                                                 train = False,
                                                 download = True,
                                                 transform = tv.transforms.ToTensor())
        elif dataset == 'cifar10':
            test_data = tv.datasets.CIFAR10(root = './data/',
                                            train = False,
                                            download = True,
                                            transform = tv.transforms.Compose([
                                                tv.transforms.CenterCrop(28),
                                                tv.transforms.ToTensor(),
                                                tv.transforms.Normalize(0.5, 0.5)
                                            ]))
        else:
            raise Exception('No other data sets!')
        data_loader = t.utils.data.DataLoader(test_data,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              drop_last = False)

    return data_loader


# Step 2 Reshape the inputs#
# Step 3 Normalize the inputs#
# Step 4 Initialize parameters#
# Step 5 Forward pass#
def get_model(in_channels = 1):
    """Get model instance
    Args :
        --in_channels: input channels, default is 1
    return :
        --model
    """
    model = CapsNet(input_channels = in_channels)
    model.to(device)
    print('Model : \n', model)

    return model


def train(dataset : str, epochs : int, batch_size : int):
    """train the model
    Args :
        --dataset: string format
        --epochs: training epochs
        --batch_size
    """
    data_loader = get_data_loader(dataset, True, batch_size = batch_size)
    model = get_model(in_channels = 3 if dataset == 'cifar10' else 1)
    # Step 6 Compute cost#
    mse = t.nn.MSELoss().to(device)

    # Step 7 Backward pass#
    optimizer = t.optim.Adam(filter(lambda x : x.requires_grad, model.parameters()),
                             amsgrad = False, weight_decay = 0)

    lr_scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.2, milestones = [20, 40, 60, 80])

    # Step 8 Update parameters#
    losses = []
    cls_losses = []
    mse_losses = []
    accs = []
    for epoch in tqdm.tqdm(range(epochs)):
        print('Epoch : %d / %d.' % (epoch + 1, epochs))
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out, out_, _ = model(batch_x, batch_y)
            batch_cls_cost = margin_loss(out, batch_y, opt.beta)
            batch_mse_cost = mse(out_, batch_x.view(batch_x.size(0), -1))
            batch_cost = 1e-3 * batch_mse_cost + batch_cls_cost

            batch_cost.backward()
            optimizer.step()

            if i % batch_size == 0:
                correct = t.argmax(out, dim = -1)
                acc = t.sum(correct == batch_y).float() / batch_x.size(0)

                print('\tBatch %d / %d has cost : %.2f[CLS : %.2f || MSE : %.2f] --> Acc : %.2f%%.' % (i + 1, len(data_loader), batch_cost.item(),
                                                                                                       batch_cls_cost.item(), batch_mse_cost.item(),
                                                                                                       acc * 100.))
                losses.append(batch_cost.item())
                cls_losses.append(batch_cls_cost.item())
                mse_losses.append(batch_mse_cost.item())
                accs.append(acc.item())

        lr_scheduler.step()

    print('Training is done!')

    f, ax = plt.subplots(1, 2, figsize = (10, 5))
    f.suptitle('Training statistics')
    ax[0].plot(range(len(losses)), losses, '-^g', label = 'loss')
    ax[0].plot(range(len(cls_losses)), cls_losses, '-*r', label = 'classification loss')
    ax[0].plot(range(len(mse_losses)), mse_losses, '-b', label = 'reconstruction loss')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Values')
    ax[0].set_title('Training loss')
    ax[0].grid(True)
    ax[0].legend(loc = 'best')

    ax[1].plot(range(len(accs)), accs, '-^y', label = 'acc')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Values')
    ax[1].set_title('Training acc')
    ax[1].grid(True)
    ax[1].legend(loc = 'best')

    plt.savefig(f'./results/training_statistics_{dataset}.png')
    plt.close()

    t.save(model, f'./results/model_{dataset}.pth')


def test(dataset : str, batch_size : int):
    """test and saving results
    Args :
        --dataset
        --batch_size
    """
    model = t.load(f'./results/model_{dataset}.pth')
    model.to(device)

    data_loader = get_data_loader(dataset, train = False, batch_size = batch_size)
    model.eval()

    string = '\n' + '*' * 40 + str(datetime.now()) + '*' * 40 + '\n'
    with t.no_grad():
        accs = []
        for i, (batch_x, batch_y) in enumerate(data_loader):
            sys.stdout.write('\r>>Testing batch %d / %d.' % (i + 1, len(data_loader)))
            sys.stdout.flush()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            out, out_, out_ori = model(batch_x)
            out_ = out_.view(-1, 3, 28, 28) if dataset == 'cifar10' else out_.view(-1, 28, 28)

            correct = t.argmax(out, dim = -1)
            batch_acc = t.sum(correct == batch_y).float() / batch_x.size(0)
            accs.append(batch_acc.item() * 100.)

            input = batch_x[:10, ...].view(-1, 3, 28, 28).cpu() if dataset == 'cifar10' else batch_x[:10, ...].view(-1, 28, 28).cpu() # input
            out_ = out_[:10, ...].cpu() # generated

            tensor = t.cat((input, out_), dim = -1)
            show_batch(tensor, i + 1, dataset)

            show_manipulation(out_ori[:1], model, index = i + 1, dataset = dataset)

    print('\nTesting is done!')

    final_acc = sum(accs) / len(data_loader)
    print('Final testing accuracy is : {:.2f}%.'.format(final_acc))

    with open(f'./results/results.txt', 'a+') as f:
        string += f'Final testing accuracy for dataset {dataset} is {final_acc}%.\n'
        f.write(string)
        f.flush()


def main(**kwargs):
    """Main function"""
    opt = Config()
    opt.parse(**kwargs)

    if 'only_test' in kwargs and kwargs['only_test']:
        test(opt.dataset, opt.batch_size)

    else:
        train(epochs = opt.epochs,
              batch_size = opt.batch_size,
              dataset = opt.dataset)

        test(opt.dataset, opt.batch_size)



if __name__ == '__main__':
    fire.Fire()

    """
    Usage:
    python main.py main --beta=0.5 --epochs=20 --batch_size=32 --only_test=False --dataset='mnist'
    """

    print('\nDone!\n')