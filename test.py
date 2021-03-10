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

    E_A.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_E_A.pth"))
    E_B.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_E_B.pth"))
    H_A.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_H_A.pth"))
    H_B.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_H_B.pth"))
    S_G_AB.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_S_G_AB.pth"))
    S_P_A.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_S_P_A.pth"))
    S_G_BA.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_S_G_BA.pth"))
    S_P_B.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_S_P_B.pth"))
    D_G_AB.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_D_G_AB.pth"))
    D_P_A.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_D_P_A.pth"))
    D_G_BA.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_D_G_BA.pth"))
    D_P_B.load_state_dict(torch.load("checkpoints_cyclegan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_D_P_B.pth"))


    # transform
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    # dataloader
    test_dataloader = DataLoader(AugGANDataset(opts.dataset_name, transform, mode='test'), batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)

    end_epoch = opts.epochs + opts.start_epoch
    total_batch = len(train_dataloader) * opts.epochs

    # test
    with torch.no_grad():
        for index, batch in enumerate(test_dataloader):
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            label_A = Variable(batch['lA'].to(device))
            label_B = Variable(batch['lB'].to(device))

            # Create fake_images, reconstructed_images and segmentation_pred_images
            fake_A = D_G_BA(S_G_BA(H_B(E_B(real_B))))
            fake_B = D_G_AB(S_G_AB(H_A(E_A(real_A))))
            rec_A = D_G_BA(S_G_BA(H_B(E_B(fake_B))))
            rec_B = D_G_AB(S_G_AB(H_A(E_A(fake_A))))
            seg_A = D_P_A(S_P_A(H_A(E_A(real_A))))
            seg_B = D_P_B(S_P_B(H_B(E_B(real_B))))


            save_image(real_A, f"result/{opts.dataset_name}/{str(index)}_A_real.png",normalize=True)
            save_image(real_B, f"result/{opts.dataset_name}/{str(index)}_B_real.png",normalize=True)
            save_image(fake_A2B, f"result/{opts.dataset_name}/{str(index)}_A2B_fake.png",normalize=True)
            save_image(fake_B2A, f"result/{opts.dataset_name}/{str(index)}_B2A_fake.png",normalize=True)
            save_image(rec_A, f"result/{opts.dataset_name}/{str(index)}_A_rec.png",normalize=True)
            save_image(rec_B, f"result/{opts.dataset_name}/{str(index)}_B_rec.png",normalize=True)
            save_image(seg_A, f"result/{opts.dataset_name}/{str(index)}_A_seg.png",normalize=True)
            save_image(seg_B, f"result/{opts.dataset_name}/{str(index)}_B_seg.png",normalize=True)
            print('calculate...'+str(index))

            

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


    os.makedirs(f"result/{opts.dataset_name}", exist_ok=True)
    print_params(opts)
    train(opts)

if __name__ == '__main__':
    main()
