import argparse
import itertools
import os
import time
import numpy as np
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn

from params import create_parser
from dataset import AugGANDataset
from models import *


def train(opts):
    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # networks
    E_A = Encoder().to(device)
    E_B = Encoder().to(device)
    H_A = HardShare().to(device)
    H_B = HardShare().to(device)
    S_G_AB = SoftShare().to(device)
    S_P_A = SoftShare().to(device)
    S_G_BA = SoftShare().to(device)
    S_P_B = SoftShare().to(device)
    D_G_AB = Decoder_Generator().to(device)
    D_P_A = Decoder_ParsingNetworks().to(device)
    D_G_BA = Decoder_Generator().to(device)
    D_P_B = Decoder_ParsingNetworks().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    #G_AB = Generator().to(device)
    #G_BA = Generator().to(device)
    #S_A = ParsingNetworks().to(device)
    #S_B = ParsingNetworks().to(device)

    # losses
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_seg_L1 = nn.L1Loss()
    criterion_seg_cross = nn.BCEWithLogitsLoss()
    criterion_identity = torch.nn.L1Loss()


    # optimizers
    optimizer_mul = torch.optim.Adam(itertools.chain(
    E_A.parameters(), H_A.parameters(),S_G_AB.parameters(),S_P_A.parameters(),D_G_AB.parameters(),D_P_A.parameters(),
    E_B.parameters(), H_B.parameters(),S_G_BA.parameters(),S_P_B.parameters(),D_G_BA.parameters(),D_P_B.parameters(),
    ),lr=opts.lr, betas=(0.5,0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opts.lr, betas=(0.5,0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opts.lr, betas=(0.5,0.999))

    # transform
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    # dataloader
    train_dataloader = DataLoader(AugGANDataset(opts.dataset_name, transform), batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    val_dataloader = DataLoader(AugGANDataset(opts.dataset_name, transform, mode='val'), batch_size=5, shuffle=True, num_workers=1)

    end_epoch = opts.epochs + opts.start_epoch
    total_batch = len(train_dataloader) * opts.epochs

    # training
    for epoch in range(opts.start_epoch, end_epoch):
        start = time.time()
        for index, batch in enumerate(train_dataloader):
            # load images and labels
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            label_A = Variable(batch['lA'].to(device))
            label_B = Variable(batch['lB'].to(device))

            # Create fake_images, reconstructed_images and segmentation_pred_images
            fake_A = D_G_BA(S_G_BA(H_B(E_B(real_B))))
            fake_B = D_G_AB(S_G_AB(H_A(E_A(real_A))))
            reconstructed_A = D_G_BA(S_G_BA(H_B(E_B(fake_B))))
            reconstructed_B = D_G_AB(S_G_AB(H_A(E_A(fake_A))))
            seg_A = D_P_A(S_P_A(H_A(E_A(real_A))))
            seg_B = D_P_B(S_P_B(H_B(E_B(real_B))))

            # backward Multitask_networks
            optimizer_mul.zero_grad()
            patch_real = D_A(real_A)
            loss_gan_BA = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_B(fake_B)
            loss_gan_AB = criterion_gan(D_B(fake_B), torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_gan = loss_gan_BA + loss_gan_AB

            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)
            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * opts.lambda_cycle # 10

            #original paper doen't use identity loss
            #loss_id_a = criterion_identity(D_G_BA(S_G_BA(H_B(E_B(real_A)))), real_A)
            #loss_id_b = criterion_identity(D_G_AB(S_G_AB(H_A(E_A(real_B)))), real_B)
            #loss_identity = (loss_id_a + loss_id_b) * 5

            loss_seg_cross_A = criterion_seg_cross(seg_A, label_A)
            loss_seg_cross_B = criterion_seg_cross(seg_B, label_B)
            loss_seg_L1_A = criterion_seg_L1(seg_A, label_A)
            loss_seg_L1_B = criterion_seg_L1(seg_B, label_B)

            loss_seg = (loss_seg_cross_A + loss_seg_cross_B + loss_seg_L1_A + loss_seg_L1_B) * opts.lambda_seg # undefined
            loss_seg = (loss_seg_cross_A + loss_seg_L1_A) * opts.lambda_seg
            # soft sharing
            dot_x = 0
            for param_g, param_p in zip(S_G_AB.parameters(), S_P_A.parameters()):
                if len(param_g.shape) != 1:
                    dot_x += torch.dot(torch.flatten(param_g),torch.flatten(param_p))
                else:
                    dot_x += torch.dot(param_g,param_p)
            l2_reg_Gx = 0
            for param in S_G_AB.parameters():
                l2_reg_Gx += torch.norm(param)
            l2_reg_Px = 0
            for param in S_P_A.parameters():
                l2_reg_Px += torch.norm(param)

            loss_weight_sharing_x = -torch.log(dot_x/(l2_reg_Gx*l2_reg_Px))

            dot_y = 0
            for param_g, param_p in zip(S_G_BA.parameters(), S_P_B.parameters()):
                if len(param_g.shape) != 1:
                    dot_y += torch.dot(torch.flatten(param_g),torch.flatten(param_p))
                else:
                    dot_y += torch.dot(param_g,param_p)
            l2_reg_Gy = 0
            for param in S_G_BA.parameters():
                l2_reg_Gy += torch.norm(param)
            l2_reg_Py = 0
            for param in S_P_B.parameters():
                l2_reg_Py += torch.norm(param)

            loss_weight_sharing_y = -torch.log(dot_y/(l2_reg_Gy*l2_reg_Py))

            loss_weight_sharing = (loss_weight_sharing_x + loss_weight_sharing_y)* opts.lambda_ws # 0.02


            # total loss
            loss_mul = loss_gan + loss_cycle + loss_seg + loss_weight_sharing #+ loss_identity
            loss_mul.backward(retain_graph=True)
            optimizer_mul.step()

            # backward Discriminator
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            fake_A = D_G_BA(S_G_BA(H_B(E_B(real_B))))
            fake_B = D_G_AB(S_G_AB(H_A(E_A(real_A))))

            # backward Discriminator A
            optimizer_D_A.zero_grad()
            patch_real = D_A(real_A)
            loss_D_A_real = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_A(fake_A)
            loss_D_A_fake = criterion_gan(patch_fake, torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_D_A = loss_D_A_real + loss_D_A_fake
            loss_D_A.backward(retain_graph=True)
            optimizer_D_A.step()

            # backward Discriminator B
            optimizer_D_B.zero_grad()
            patch_real = D_B(real_B)
            loss_D_B_real = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_B(fake_B)
            loss_D_B_fake = criterion_gan(patch_fake, torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_D_B = loss_D_B_real + loss_D_B_fake
            loss_D_B.backward()
            optimizer_D_B.step()

            if index % 10 == 0:
                print(f"\r[Epoch {epoch+1}/{opts.epochs-opts.start_epoch}] [Index {index}/{len(train_dataloader)}] [D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] [G loss: adv: {loss_gan.item():.4f}, loss_cycle: {loss_cycle.item():.4f}, loss_seg: {loss_seg.item():.4f}, loss_weight_sharing: {loss_weight_sharing.item():.4f}]")



        # save samples
        save_sample(E_A, E_B, H_A, H_B, S_G_BA, S_G_AB, D_G_BA, D_G_AB, S_P_A, S_P_B, D_P_A, D_P_B, epoch+1, opts, val_dataloader)

        if epoch % opts.checkpoint_every == 0:
            torch.save(E_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_E_B.pth')
            torch.save(E_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_E_A.pth')
            torch.save(H_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_H_B.pth')
            torch.save(H_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_H_A.pth')
            torch.save(S_G_BA.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_S_G_BA.pth')
            torch.save(S_G_AB.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_S_G_AB.pth')
            torch.save(S_P_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_S_P_A.pth')
            torch.save(S_P_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_S_P_B.pth')
            torch.save(D_G_BA.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_D_G_BA.pth')
            torch.save(D_G_AB.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_D_G_AB.pth')
            torch.save(D_P_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_D_P_A.pth')
            torch.save(D_P_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_D_P_B.pth')
            torch.save(E_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_E_B.pth')
            torch.save(E_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_E_A.pth')
            torch.save(H_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_H_B.pth')
            torch.save(H_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_H_A.pth')
            torch.save(S_G_BA.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_S_G_BA.pth')
            torch.save(S_G_AB.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_S_G_AB.pth')
            torch.save(S_P_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_S_P_A.pth')
            torch.save(S_P_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_S_P_B.pth')
            torch.save(D_G_BA.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_D_G_BA.pth')
            torch.save(D_G_AB.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_D_G_AB.pth')
            torch.save(D_P_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_D_P_A.pth')
            torch.save(D_P_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_D_P_B.pth')
        print(f"[Epoch {epoch+1} : {time.time() - start} sec")

def save_sample(E_A, E_B, H_A, H_B, S_G_BA, S_G_AB, D_G_BA, D_G_AB, S_P_A, S_P_B, D_P_A, D_P_B, epoch, opts, val_dataloader):
    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')
    images = next(iter(val_dataloader))
    real_A = Variable(images['A'].to(device))
    real_B = Variable(images['B'].to(device))
    fake_A = D_G_BA(S_G_BA(H_B(E_B(real_B))))
    fake_B = D_G_AB(S_G_AB(H_A(E_A(real_A))))
    seg_A = D_P_A(S_P_A(H_A(E_A(real_A))))
    seg_B = D_P_B(S_P_B(H_B(E_B(real_B))))
    rec_A = D_G_BA(S_G_BA(H_B(E_B(fake_B))))
    rec_B = D_G_AB(S_G_AB(H_A(E_A(fake_A))))

    real_A = ((real_A-torch.min(real_A))/(torch.max(real_A)-torch.min(real_A)))*255
    real_B = ((real_B-torch.min(real_B))/(torch.max(real_B)-torch.min(real_B)))*255
    fake_A = ((fake_A-torch.min(fake_A))/(torch.max(fake_A)-torch.min(fake_A)))*255
    fake_B = ((fake_B-torch.min(fake_B))/(torch.max(fake_B)-torch.min(fake_B)))*255
    rec_A = ((rec_A-torch.min(rec_A))/(torch.max(rec_A)-torch.min(rec_A)))*255
    rec_B = ((rec_B-torch.min(rec_B))/(torch.max(rec_B)-torch.min(rec_B)))*255


    N, _, h, w = seg_A.shape
    seg_A = seg_A.reshape(N,20, -1).argmax(axis=1).reshape(N, 1, h, w).repeat(1, 3, 1, 1)*10
    seg_B = seg_B.reshape(N,20, -1).argmax(axis=1).reshape(N, 1, h, w).repeat(1, 3, 1, 1)*10


    image_sample = torch.cat((real_A.data, fake_B.data, seg_A.data, rec_A.data,
                              real_B.data, fake_A.data, seg_B.data, rec_B.data
                              ), 0)
    save_image(image_sample, f"{opts.samples_cyclegan_dir}/{opts.dataset_name}/{epoch}.png", nrow=5,normalize=True)


def print_params(opts):
    print('=' * 80)
    print('Params'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

def main():
    parser = create_parser()
    opts = parser.parse_args()

    os.makedirs(f"{opts.sample_dir}/{opts.dataset_name}", exist_ok=True)
    os.makedirs(f"{opts.checkpoint_dir}/{opts.dataset_name}", exist_ok=True)
    os.makedirs(f"{opts.samples_cyclegan_dir}/{opts.dataset_name}", exist_ok=True)
    os.makedirs(f"{opts.samples_seg_dir}/{opts.dataset_name}", exist_ok=True)
    print_params(opts)
    train(opts)

if __name__ == '__main__':
    main()
