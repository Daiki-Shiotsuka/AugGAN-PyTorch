import argparse
def create_parser():

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--image_height', type=int, default=256, help='image_height.')
    parser.add_argument('--image_width', type=int, default=256, help='image_width.')
    parser.add_argument('--a_channels', type=int, default=3, help='num_A_channels.')
    parser.add_argument('--b_channels', type=int, default=3, help='num_B_channels.')
    parser.add_argument('--dataset_name', type=str, default='train_data', help='dataset_name.')
    parser.add_argument('--epochs', type=int, default=200, help='num_Epochs.')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size.')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers.')
    parser.add_argument('--lr', type=float, default=0.0002, help='lr_param.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam_param1.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam_param1.')
    parser.add_argument('--n_cpu', type=int, default=8, help='n_cpu.')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id.')
    parser.add_argument('--lambda_cycle', type=int, default=10, help='lambda_cycle.')
    parser.add_argument('--lambda_seg', type=float, default=0.2, help='lambda_seg.')
    parser.add_argument('--lambda_ws', type=float, default=0.02, help='lambda_ws.')
    parser.add_argument('--seg_channels', type=int, default=20, help='seg_channels.')
    parser.add_argument('--test_epoch', type=str, default='latest', help='test_epoch')
    parser.add_argument('--dataroot_dir', type=str, default='../data/')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--samples_seg_dir', type=str, default='samples_seg')
    parser.add_argument('--samples_cyclegan_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=20)
    parser.add_argument('--checkpoint_every', type=int , default=1, help='checkpoint_every')
    return parser
