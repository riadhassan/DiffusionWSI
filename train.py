import os
import argparse
from utils.utils import *
from DataLoader.data_loader import get_dataloader
from omegaconf import OmegaConf
from trainer.demo_trainer import DemoTrainer




def main(args):

    # loads configs
    configs = OmegaConf.load(args.config_file)

    # initialize environments
    init_environment(configs.seed)
    exp_dir = os.path.join(args.exp_root, configs.exp)

    # prints information
    print('-' * 100)
    print('Training for BCI Dataset ...\n')
    print(f'- Train Dir: {args.train_dir}')
    print(f'-  Val  Dir: {args.val_dir}')
    print(f'-  Exp  Dir: {exp_dir}')
    print(f'- Configs  : {args.config_file}')
    print(f'- Trainer  : {args.trainer}', '\n')

    if args.trainer == 'basic':
        # loads dataloder for training and validation
        train_loader = get_dataloader('train', args.train_dir, configs.loader)
        val_loader   = get_dataloader('val',   args.val_dir,   configs.loader)

        # initialize trainer
        trainer = DemoTrainer(configs, exp_dir, args.resume_ckpt)

    # elif args.trainer == 'cahr':
    #     # loads dataloder for training and validation
    #     train_loader = get_cahr_dataloader('train', args.train_dir, configs.loader)
    #     val_loader   = get_cahr_dataloader('val',   args.val_dir,   configs.loader)
    #
    #     # initialize trainer
    #     trainer = BCITrainerCAHR(configs, exp_dir, args.resume_ckpt)

    # training model
    trainer.forward(train_loader, val_loader)

    print('-' * 100, '\n')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training for BCI Dataset')
    parser.add_argument('--train_dir',   type=str, default="data\\actual\\train", help='dir path of training data')
    parser.add_argument('--val_dir',     type=str, default="data\\actual\\val", help='dir path of validation data')
    parser.add_argument('--exp_root',    type=str, help='root dir of experiment', default="experiment")
    parser.add_argument('--config_file', type=str, default="configs\\exp_config.yml", help='yaml path of configs')
    parser.add_argument('--resume_ckpt', type=str, help='checkpoint path for resuming')
    parser.add_argument('--trainer',     type=str, help='trainer type, basic or cahr', default='basic')
    args = parser.parse_args()

    check_train_args(args)
    main(args)