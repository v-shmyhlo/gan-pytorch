import argparse
import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from ticpfptp.format import args_to_string
from ticpfptp.metrics import Mean
from ticpfptp.torch import fix_seed
from tqdm import tqdm

import utils
from discriminator import Conv as ConvDiscriminator
from generator import Conv as ConvGenerator

# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z

NUM_CHANNELS = 3


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--latent-size', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    fix_seed(args.seed)

    transform = T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    data_loader = torch.utils.data.DataLoader(
        # torchvision.datasets.MNIST(args.dataset_path, transform=transform, download=True),
        torchvision.datasets.ImageFolder(args.dataset_path, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discriminator = ConvDiscriminator(NUM_CHANNELS)
    generator = ConvGenerator(args.latent_size, NUM_CHANNELS)
    discriminator.to(device)
    generator.to(device)

    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    generator_opt = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    noise_dist = torch.distributions.Normal(0, 1)

    writer = SummaryWriter(args.experiment_path)
    metrics = {
        'loss/discriminator': Mean(),
        'loss/generator': Mean()
    }

    for epoch in range(args.epochs):
        discriminator.train()
        generator.train()
        for real, _ in tqdm(data_loader, desc='epoch {} training'.format(epoch)):
            real = real.to(device)

            # discriminator
            discriminator_opt.zero_grad()

            # real
            logits = discriminator(real)
            loss = F.binary_cross_entropy_with_logits(input=logits, target=torch.ones_like(logits).to(device))
            loss.mean().backward()
            loss_real = loss

            # fake
            noise = noise_dist.sample((args.batch_size, args.latent_size)).to(device)
            fake = generator(noise)
            logits = discriminator(fake)
            loss = F.binary_cross_entropy_with_logits(input=logits, target=torch.zeros_like(logits).to(device))
            loss.mean().backward()
            loss_fake = loss

            discriminator_opt.step()
            metrics['loss/discriminator'].update((loss_real + loss_fake).data.cpu().numpy())

            # generator
            generator_opt.zero_grad()

            # fake
            noise = noise_dist.sample((args.batch_size, args.latent_size)).to(device)
            fake = generator(noise)
            logits = discriminator(fake)
            loss = F.binary_cross_entropy_with_logits(input=logits, target=torch.ones_like(logits).to(device))

            loss.mean().backward()
            generator_opt.step()
            metrics['loss/generator'].update(loss.data.cpu().numpy())

        writer.add_scalar('loss/discriminator', metrics['loss/discriminator'].compute_and_reset(), global_step=epoch)
        writer.add_scalar('loss/generator', metrics['loss/generator'].compute_and_reset(), global_step=epoch)
        writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)


if __name__ == '__main__':
    main()
