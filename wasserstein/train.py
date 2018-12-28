import argparse
import os
import torchvision
from ticpfptp.metrics import Mean
from dataset import Dataset
import torch.utils.data
import torch
import logging
from tqdm import tqdm
from ticpfptp.format import args_to_string
from ticpfptp.torch import fix_seed, save_model, load_weights
from discriminator import Convolutional as ConvolutionalDiscriminator
from generator import Convolutional as ConvolutionalGenerator
from tensorboardX import SummaryWriter


# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--latent-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--discr-steps', type=int, default=5)
    parser.add_argument('--discr-clamp', type=float, nargs=2, default=[-0.01, 0.01])
    # parser.add_argument('--opt', type=str, choices=['adam', 'momentum'], default='momentum')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    # experiment_path = os.path.join(args.experiment_path, args_to_path(
    #     args, ignore=['experiment_path', 'restore_path', 'dataset_path', 'epochs', 'workers']))
    fix_seed(args.seed)

    data_loader = torch.utils.data.DataLoader(
        Dataset(args.dataset_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discriminator = ConvolutionalDiscriminator(args.latent_size)
    generator = ConvolutionalGenerator(args.latent_size)

    discriminator.to(device)
    generator.to(device)

    dist = torch.distributions.Normal(0, 1)

    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    generator_opt = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

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
                for p in discriminator.parameters():
                    p.data.clamp_(args.discr_clamp[0], args.discr_clamp[1])

                discriminator_opt.zero_grad()

                # real
                real, _ = next(data_loader_iter)
                real = real.to(device)
                score = discriminator(real)
                score.mean().backward()
                metrics['score/real'].update(score.data.cpu().numpy())
                score_real = score

                # fake
                noise = dist.sample((args.batch_size, args.latent_size)).to(device)
                # noise = noise / noise.norm(dim=-1, keepdim=True)
                fake = generator(noise)
                score = discriminator(fake)
                (-score.mean()).backward()
                metrics['score/fake'].update(score.data.cpu().numpy())
                score_fake = score

                discriminator_opt.step()
                metrics['score/delta'].update((score_real - score_fake).data.cpu().numpy())

            # generator
            noise = dist.sample((args.batch_size, args.latent_size)).to(device)
            # noise = noise / noise.norm(dim=-1, keepdim=True)
            fake = generator(noise)
            score = discriminator(fake)

            generator_opt.zero_grad()
            score.mean().backward()
            generator_opt.step()

        writer.add_scalar('score/real', metrics['score/real'].compute_and_reset(), global_step=epoch)
        writer.add_scalar('score/fake', metrics['score/fake'].compute_and_reset(), global_step=epoch)
        writer.add_scalar('score/delta', metrics['score/delta'].compute_and_reset(), global_step=epoch)
        writer.add_image('real', torchvision.utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', torchvision.utils.make_grid((fake + 1) / 2), global_step=epoch)


if __name__ == '__main__':
    main()
