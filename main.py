from network import SRDenseNet
from utils import *
import argparse

def check_phase(args):
    args.phase = args.phase.lower()
    assert args.phase in ['train', 'test', 'infer'], 'Choose Train/Test/Infer phase'

def check_args(args):
    check_phase()
    return args

def parse_args():
    desc = "Tensorflow implementation of SRDenseNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='Choose Train/Test/Infer Phase.')

    parser.add_argument('--channel', type=str, default='Y', help='RGB or Y or YCbCr')
    parser.add_argument('--scale', type=int, default=4, help='Super-Resolution Scale')

    parser.add_argument('--epoch', type=int, default=120, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')

    parser.add_argument('--val_interval', type=int, default=100, help='The validation interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--infer_dir', type=str, default='inferred', help='Directory name to save the infered images')
    return check_args(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    print(args)
    # quit()
    with tf.Session() as sess:
    
        cnn = SRDenseNet(sess, args)
        cnn.build_model()
        show_all_variables()
    
        if args.phase == 'train':
            cnn.train()
            cnn.test()
        elif args.phase == 'test':
            cnn.test()
        elif args.phase == 'infer':
            cnn.infer()