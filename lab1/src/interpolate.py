import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import autoencoderMLP4Layer

DATA_DIR = './data'   # change if using Colab

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def first_index_of(test_set, class_id):
    # first test image belonging to the given class
    return (test_set.targets == class_id).nonzero()[0].item()


def prep(test_set, idx, N_input, device):
    # normalize and flatten, as in test.py
    img = test_set.data[idx].type(torch.float32) / 255.0
    return img.view(1, N_input).to(device=device)


def main():
    print('running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [8]')
    argParser.add_argument('-n', metavar='steps', type=int, help='# of interpolation steps [8]')
    argParser.add_argument('-i', metavar='index 1', type=int, help='test index of first image')
    argParser.add_argument('-j', metavar='index 2', type=int, help='test index of second image')
    argParser.add_argument('-p', metavar='plot', type=str, help='output plot file (.png)')

    args = argParser.parse_args()

    save_file = args.s
    bottleneck_size = args.z
    n_steps = args.n if args.n != None else 8
    plot_file = args.p if args.p != None else 'interpolate.MLP.' + str(bottleneck_size) + '.png'

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = FashionMNIST(DATA_DIR, train=False, download=False, transform=test_transform)

    # two images from two different classes
    idx1 = args.i if args.i != None else first_index_of(test_set, 1)   # Trouser
    idx2 = args.j if args.j != None else first_index_of(test_set, 9)   # Ankle boot

    label1 = test_set.targets[idx1].item()
    label2 = test_set.targets[idx2].item()
    print('\t\tfrom index', idx1, '(' + CLASSES[label1] + ')')
    print('\t\t  to index', idx2, '(' + CLASSES[label2] + ')')
    if label1 == label2:
        print('\t\twarning: both images are the same class')

    N_input = 28 * 28
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.to(device)
    model.eval()

    img1 = prep(test_set, idx1, N_input, device)
    img2 = prep(test_set, idx2, N_input, device)

    with torch.no_grad():
        z1 = model.encode(img1)                                  # (1, bottleneck)
        z2 = model.encode(img2)

        # n evenly spaced weights from 0 to 1, broadcast against the bottlenecks:
        # z = (1 - a) * z1 + a * z2
        a = torch.linspace(0.0, 1.0, n_steps, device=device).view(n_steps, 1)
        z = (1.0 - a) * z1 + a * z2                              # (n, bottleneck)

        middle = model.decode(z)                                 # (n, 784)
        left = model.decode(z1)
        right = model.decode(z2)

    row = [left.view(28, 28).cpu()]
    row += [middle[k].view(28, 28).cpu() for k in range(n_steps)]
    row += [right.view(28, 28).cpu()]

    fig, axes = plt.subplots(1, len(row), figsize=(len(row) * 1.2, 1.8))
    for ax, im in zip(axes, row):
        ax.imshow(im, cmap='gray')
        ax.axis('off')
    axes[0].set_title(CLASSES[label1], fontsize=8)
    axes[-1].set_title(CLASSES[label2], fontsize=8)
    plt.tight_layout()
    print('saving ', plot_file)
    plt.savefig(plot_file)
    plt.show()


if __name__ == '__main__':
    main()
