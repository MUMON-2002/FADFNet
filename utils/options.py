import argparse
import os


class TrainOptions():

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--gpu_id', type=int, required=True, help='which gpu to use')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
        self.parser.add_argument('--total_epoch', type=int, default=30, help='Total number of epochs')
        self.parser.add_argument('--warmup_epoch', type=int, default=0, help='Number of warmup epochs within total epochs in the first')

        self.parser.add_argument('--continue_to_train', action='store_true', help='Continue any interrupted training')
        self.parser.add_argument('--path_to_save', type=str, required=True, help='Path to save the trained model')
        self.parser.add_argument('--ckpt_path', type=str, default='-', help='Path to trained and saved checkpoint model')
        self.parser.add_argument('--validation_freq', type=int, default=5, help='Frequency to run validation')
        self.parser.add_argument('--save_freq', type=int, default=5, help='Frequency to save model')
        self.parser.add_argument('--use_dataset', type=str, default="mayo16", help='type of dataset')
        self.parser.add_argument('--train_ratio', type=float, default=0.9, help='Train ratio of the dataset, rest for validation')

        self.parser.add_argument('--dim', type=int, default=64, help='Transformer block dimension')
        self.parser.add_argument('--which_model', type=str, default="FADFNet", help='which type of the model to use')
        self.parser.add_argument('--norm_type', type=str, default="GN", help='which normalization type for the model to use, BN or GN or None')
        self.parser.add_argument('--activation', type=str, default="ReLU", help='which activation type for the model to use, ReLU or LeakyReLU or GELU')

        self.parser.add_argument('--wave_type', type=str, default="haar", help='wave type for wavelet transform')
        self.parser.add_argument('--wave_level', type=int, default=2, help='level of wavelet transform')


        self.initialized = True

    def parse(self):

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')

        for k, v in sorted(args.items()):
            print(f'{k}: {v}')

        print('-------------- End ----------------')

        return self.opt


class TestOptions():

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--ckpt_path', type=str, required = True, help='Path to trained and saved checkpoint model')
        self.parser.add_argument('--output_root', type=str, required = True, help='Path to save denoised imgs')
        self.parser.add_argument('--use_dataset', type=str, default="mayo16", help='type of dataset')

        self.parser.add_argument('--dim', type=int, default=64, help='Transformer block dimension')
        self.parser.add_argument('--norm_type', type=str, default="GN", help='which normalization type for the model to use, BN or GN or None')
        self.parser.add_argument('--activation', type=str, default="ReLU", help='which activation type for the model to use, ReLU or LeakyReLU or GELU')

        self.parser.add_argument('--which_model', type=str, default="FADFNet", help='which type of the model to use')
        self.parser.add_argument('--gpu_id', type=int, required=True, help='which gpu to use')

        self.parser.add_argument('--wave_type', type=str, default="haar", help='wave type for wavelet transorm')
        self.parser.add_argument('--wave_level', type=int, default=2, help='level of wavelet transorm')

        self.initialized = True

    def parse(self):

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')

        for k, v in sorted(args.items()):
            print(f'{k}: {v}')

        print('-------------- End ----------------')

        return self.opt
