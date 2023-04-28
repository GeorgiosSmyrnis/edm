# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import json

from tqdm import tqdm

from torchmetrics import StructuralSimilarityIndexMeasure

from torch.utils.data import DataLoader
from torch_utils.misc import AverageMeter
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, inputs, projection_to_measurements, mask, class_labels=None,
    randn_like=torch.randn_like, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, self_cond=False,
    dps=False, likelihood_step_size=1.0, meas_cond=False
):
    # Adjust noise levels based on what's supported by the network.
    if hasattr(net, "backbone"):
        sigma_min = max(sigma_min, net.backbone.sigma_min)
        sigma_max = min(sigma_max, net.backbone.sigma_max)
    else:
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=inputs.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    if hasattr(net, "backbone"):
        t_steps = torch.cat([net.backbone.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    else:
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    latents = inputs[:, :3, ...].to(torch.float64) * t_steps[0]
    measurements = inputs[:, 3:, ...].to(torch.float64)

    # Main sampling loop.
    x_next = latents
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        if hasattr(net, "backbone"):
            t_hat = net.backbone.round_sigma(t_cur + gamma * t_cur)
        else:
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        if self_cond:
            
            # DDRM
            with torch.no_grad():
                denoised_one_step = net(torch.cat([x_hat, torch.zeros_like(x_hat)], dim=1), t_hat, class_labels)[:, :3, ...].to(torch.float64)
            projected_measurements = projection_to_measurements(denoised_one_step, measurements)   
            if dps:
                x_hat = x_hat.requires_grad() #starting grad tracking with the noised img
            denoised = net(torch.cat([x_hat, projected_measurements], dim=1), t_hat, class_labels).to(torch.float64)[:, :3, ...] 
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if dps:
                Ax = mask * denoised
                residual = measurements - Ax
                sse = torch.sum(torch.square(residual))
                likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
                x_next = x_next - (likelihood_step_size / sse) * likelihood_score
                x_next = x_next.detach()
            else:
                y_noisy = measurements + (t_next - t_hat) * d_cur
                x_next = torch.logical_not(mask) * x_next + mask * y_noisy        
            # if i < num_steps - 1:
            #     denoised = projection_to_measurements(denoised, measurements)
            # denoised = torch.clamp(denoised, -1.0, 1.0)




            # with torch.no_grad():
            #     denoised_one_step = net(torch.cat([x_hat, torch.zeros_like(x_hat)], dim=1), t_hat, class_labels)[:, :3, ...].to(torch.float64)
            # projected_measurements = projection_to_measurements(denoised_one_step, measurements)
            # denoised = net(torch.cat([x_hat, projected_measurements], dim=1), t_hat, class_labels).to(torch.float64)[:, :3, ...]
            # if i < num_steps - 1:
            #     denoised = projection_to_measurements(denoised, measurements)
            # denoised = torch.clamp(denoised, -1.0, 1.0)
            # d_cur = (x_hat - denoised) / t_hat
            # x_next = x_hat + (t_next - t_hat) * d_cur

            # # if i < num_steps - 1:
            # #     projected_measurements = projection_to_measurements(x_next, measurements)
            # #     denoised = net(torch.cat([x_next, projected_measurements], dim=1), t_next, class_labels).to(torch.float64)[:, :3, ...]
            # #     d_prime = (x_next - denoised) / t_next
            # #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        else:
            if dps:
                x_hat = x_hat.requires_grad() #starting grad tracking with the noised img

            # Euler step.
            if meas_cond:
                net_input = torch.cat([x_hat, measurements], dim=1)
            else:
                net_input = x_hat
            denoised = net(net_input, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            
            if dps:
                Ax = mask * denoised
                residual = measurements - Ax
                sse = torch.sum(torch.square(residual))
                likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
                x_next = x_next - (likelihood_step_size / sse) * likelihood_score
                x_next = x_next.detach()

            # Apply 2nd order correction.
            if i < num_steps - 1 and not dps:
                denoised = net(torch.cat([x_next, measurements], dim=1), t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--dataset',                 help='Dataset to perform denoising.',                                    type=str, default='cifar10')
@click.option('--data_dir',                help='Dataset directory.',                                               type=str, required=True)

@click.option('--self_cond',               help='Self conditioning',                                                is_flag=True)

@click.option('--dps',                     help='Whether to use Diffusion Posterior Sampling (DPS)',                is_flag=True)
@click.option('--likelihood_step_size',    help='log-likelihood gradient step size for DPS', metavar='FLOAT',       type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)

@click.option('--meas_cond',               help='Whether the network takes measurements as conditioning input',     is_flag=True)

def main(network_pkl, dataset, data_dir, outdir, subdirs, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """

    if dataset == 'cifar10':
        normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = Compose([
            ToTensor(),
            normalize
        ])
        dataset = CIFAR10(data_dir, train=False, download=True, transform=transform)
        dataloader = DataLoader(dataset, max_batch_size, shuffle=False, num_workers=96, drop_last=False)
    else:
        raise ValueError("Dataset not supported.")


    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=True) as f:
        net = pickle.load(f)['ema'].to(device)

    # Save run options.
    opts = dnnlib.EasyDict(sampler_kwargs)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'options.json'), 'wt') as f:
        json.dump(opts, f, indent=2)

    # Loop over batches.
    print(f'Generating {len(dataset)} images to "{outdir}"...')

    mae_loss = torch.nn.L1Loss()
    avg_mae = AverageMeter()

    ssim = StructuralSimilarityIndexMeasure()
    avg_ssim = AverageMeter()

    for i, (images, labels) in tqdm(enumerate(dataloader)):

        images = images.cuda()
        labels = labels.cuda()

        threshold = 0.4 * torch.rand((1,)).to(device) + 0.1  #Between 10-50% pixels dropped
        mask = (torch.rand_like(images[:, [0], ...]) > threshold)
        # mask = torch.zeros_like(images[:, [0], ...])
        # mask[:, :, :, :16] = 1
        # mask = mask.bool()
        measurements = images * mask
        projection_to_measurements = lambda x, y: torch.logical_not(mask) * x + y

        batch_seeds = torch.randint(0, 100000, (images.shape[0],))

        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn_like(images)
        latents = torch.cat([latents, measurements], dim=1)
        class_labels = None
        if (hasattr(net, "label_dim") and net.label_dim):
            class_labels = torch.eye(net.label_dim, device=device)[labels]
        elif (hasattr(net.backbone, "label_dim") and net.backbone.label_dim):
            class_labels = torch.eye(net.backbone.label_dim, device=device)[labels]

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        recon_images = sampler_fn(net, latents, projection_to_measurements, mask, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        with torch.no_grad():
            loss = mae_loss(images, recon_images)
            avg_mae.update(loss, n=images.shape[0])

            metric = ssim(((images + 1.0) / 2).to(torch.float64), (recon_images + 1.0) / 2)
            avg_ssim.update(metric, n=images.shape[0])

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        recon_images_np = (recon_images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for recon_image_np, image_np in zip(recon_images_np, images_np):
            image_dir = outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{i:06d}a.png')
            recon_image_path = os.path.join(image_dir, f'{i:06d}b.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                PIL.Image.fromarray(recon_image_np[:, :, 0], 'L').save(recon_image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                PIL.Image.fromarray(recon_image_np, 'RGB').save(recon_image_path)

    # Done.
    print('Done.')
    print(f'Average MAE: {avg_mae.get():.4f}')
    print(f'Average SSIM: {avg_ssim.get()}')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
