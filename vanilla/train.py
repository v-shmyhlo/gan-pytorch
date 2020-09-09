import os

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from tensorboardX import SummaryWriter
from ticpfptp.metrics import Mean
from tqdm import tqdm

import utils
from discriminator import Conv as ConvDiscriminator
from generator import Conv as ConvGenerator

# TODO: spherical z
# TODO: spherical interpolation
# TODO: norm z


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)

    transform = T.Compose([
        T.Resize(config.image_size),
        T.CenterCrop(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    if config.dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(config.dataset_path, transform=transform, download=True)
        NUM_CHANNELS = 1
    elif config.dataset == 'celeba':
        dataset = torchvision.datasets.ImageFolder(config.dataset_path, transform=transform)
        NUM_CHANNELS = 3
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True)

    model = nn.ModuleDict({
        'discriminator': ConvDiscriminator(
            config.image_size, NUM_CHANNELS, base_features=config.model.base_features),
        'generator': ConvGenerator(
            config.image_size, config.latent_size, NUM_CHANNELS, base_features=config.model.base_features),
    })
    model.to(DEVICE)
    model.apply(weights_init)
    if config.restore_path is not None:
        model.load_state_dict(torch.load(config.restore_path))

    discriminator_opt = torch.optim.Adam(
        model.discriminator.parameters(), lr=config.opt.lr, betas=(0.5, 0.999))
    generator_opt = torch.optim.Adam(
        model.generator.parameters(), lr=config.opt.lr, betas=(0.5, 0.999))

    noise_dist = torch.distributions.Normal(0, 1)

    writer = SummaryWriter(config.experiment_path)
    metrics = {
        'loss/discriminator': Mean(),
        'loss/generator': Mean()
    }

    for epoch in range(1, config.epochs + 1):
        model.train()
        for real, _ in tqdm(data_loader, desc='epoch {} training'.format(epoch)):
            real = real.to(DEVICE)

            # discriminator ############################################################################################
            discriminator_opt.zero_grad()

            # real
            scores = model.discriminator(real)
            loss = F.softplus(-scores)
            loss.mean().backward()
            loss_real = loss

            # fake
            noise = noise_dist.sample((config.batch_size, config.latent_size)).to(DEVICE)
            fake = model.generator(noise)
            scores = model.discriminator(fake)
            loss = F.softplus(scores)
            loss.mean().backward()
            loss_fake = loss

            discriminator_opt.step()
            metrics['loss/discriminator'].update((loss_real + loss_fake).data.cpu().numpy())

            # generator ################################################################################################
            generator_opt.zero_grad()

            # fake
            noise = noise_dist.sample((config.batch_size, config.latent_size)).to(DEVICE)
            fake = model.generator(noise)
            scores = model.discriminator(fake)
            loss = F.softplus(-scores)
            loss.mean().backward()

            generator_opt.step()
            metrics['loss/generator'].update(loss.data.cpu().numpy())

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('real', utils.make_grid((real + 1) / 2), global_step=epoch)
        writer.add_image('fake', utils.make_grid((fake + 1) / 2), global_step=epoch)

        torch.save(model.state_dict(), os.path.join(config.experiment_path, 'model.pth'))


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight.data, 0., 0.02)
    elif isinstance(m, (nn.BatchNorm2d,)):
        torch.nn.init.normal_(m.weight.data, 1., 0.02)
        torch.nn.init.constant_(m.bias.data, 0.)


if __name__ == '__main__':
    main()
