import argparse
import logging
import os

import torch
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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--latent-size', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--discr-steps', type=int, default=5)
    parser.add_argument('--discr-clamp', type=float, default=0.01)
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
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.dataset_path, transform=transform, download=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discriminator = ConvDiscriminator(1)
    generator = ConvGenerator(args.latent_size, 1)
    discriminator.to(device)
    generator.to(device)

    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    generator_opt = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    noise_dist = torch.distributions.Normal(0, 1)

    writer = SummaryWriter(args.experiment_path)
    metrics = {
        'score/real': Mean(),
        'score/fake': Mean(),
        'score/delta': Mean()
    }

    for epoch in range(args.epochs):
        data_loader_iter = iter(data_loader)

        discriminator.train()
        generator.train()
        for _ in tqdm(range(len(data_loader) // args.discr_steps), desc='epoch {} training'.format(epoch)):
            # discriminator
            for _ in range(args.discr_steps):
                discriminator_opt.zero_grad()

                # real
                real, _ = next(data_loader_iter)
                real = real.to(device)
                score = discriminator(real)
                score.mean().backward()
                metrics['score/real'].update(score.data.cpu().numpy())
                score_real = score

                # fake
                noise = noise_dist.sample((args.batch_size, args.latent_size)).to(device)
                fake = generator(noise)
                score = discriminator(fake)
                (-score.mean()).backward()
                metrics['score/fake'].update(score.data.cpu().numpy())
                score_fake = score

                discriminator_opt.step()
                metrics['score/delta'].update((score_real - score_fake).data.cpu().numpy())

                for p in discriminator.parameters():
                    p.data.clamp_(-args.discr_clamp, args.discr_clamp)
                   
            # generator
            noise = noise_dist.sample((args.batch_size, args.latent_size)).to(device)
            fake = generator(noise)
            score = discriminator(fake)

            generator_opt.zero_grad()
            score.mean().backward()
            generator_opt.step()

        writer.add_scalar('score/real', metrics['score/real'].compute_and_reset(), global_step=epoch)
        writer.add_scalar('score/fake', metrics['score/fake'].compute_and_reset(), global_step=epoch)
        writer.add_scalar('score/delta', metrics['score/delta'].compute_and_reset(), global_step=epoch)
        writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)


if __name__ == '__main__':
    main()
