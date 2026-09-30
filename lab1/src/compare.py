import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from model import autoencoderMLP4Layer

DATA_DIR = './data'   # change if using Colab

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# bottleneck size -> parameter file, as trained in step 1
MODELS = [(8, 'MLP.8.pth'), (32, 'MLP.32.pth'), (128, 'MLP.128.pth')]


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
    argParser.add_argument('-i', metavar='indices', type=str,
                           help='comma separated test indices [one per class]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output plot file (.png)')

    args = argParser.parse_args()

    plot_file = args.p if args.p != None else 'compare.bottlenecks.png'

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = FashionMNIST(DATA_DIR, train=False, download=False, transform=test_transform)

    if args.i != None:
        indices = [int(s) for s in args.i.split(',')]
    else:
        # one image from each of the ten classes
        indices = [first_index_of(test_set, c) for c in range(len(CLASSES))]
    print('\t\tindices = ', indices)

    N_input = 28 * 28
    N_output = N_input

    imgs = torch.cat([prep(test_set, idx, N_input, device) for idx in indices])

    # one row of reconstructions per model, all from the same test images
    rows = [('input', imgs)]
    for bottleneck_size, save_file in MODELS:
        model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size,
                                     N_output=N_output)
        model.load_state_dict(torch.load(save_file, weights_only=True))
        model.to(device)
        model.eval()

        with torch.no_grad():
            rows += [('z = ' + str(bottleneck_size), model(imgs))]

    fig, axes = plt.subplots(len(rows), len(indices),
                             figsize=(len(indices) * 1.2, len(rows) * 1.4))
    for r, (name, batch) in enumerate(rows):
        for c, idx in enumerate(indices):
            ax = axes[r][c]
            ax.imshow(batch[c].view(28, 28).cpu(), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(CLASSES[test_set.targets[idx].item()], fontsize=7)
        axes[r][0].set_ylabel(name, fontsize=8)

    plt.tight_layout()
    print('saving ', plot_file)
    plt.savefig(plot_file)
    plt.show()


if __name__ == '__main__':
    main()
