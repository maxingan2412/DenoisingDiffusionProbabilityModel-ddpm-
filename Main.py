import argparse
from Diffusion.Train import train, eval


def main(args):
    print("Running with the following args:", args)  # 打印所有的arg参数

    modelConfig = {
        "state": args.state,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "T": args.T,
        "channel": args.channel,
        "channel_mult": args.channel_mult,
        "attn": args.attn,
        "num_res_blocks": args.num_res_blocks,
        "dropout": args.dropout,
        "lr": args.lr,
        "multiplier": args.multiplier,
        "beta_1": args.beta_1,
        "beta_T": args.beta_T,
        "img_size": args.img_size,
        "grad_clip": args.grad_clip,
        "device": args.device,
        "training_load_weight": args.training_load_weight,
        "save_weight_dir": args.save_weight_dir,
        "test_load_weight": args.test_load_weight,
        "sampled_dir": args.sampled_dir,
        "sampledNoisyImgName": args.sampledNoisyImgName,
        "sampledImgName": args.sampledImgName,
        "nrow": args.nrow
    }

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion model training and evaluation.')
    parser.add_argument('--state', type=str, default='train', help='State: train or eval')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size.')
    parser.add_argument('--T', type=int, default=1000, help='T parameter.')
    parser.add_argument('--channel', type=int, default=128, help='Channel size.')
    parser.add_argument('--channel_mult', nargs='+', type=int, default=[1, 2, 3, 4], help='Channel multipliers.')
    parser.add_argument('--attn', nargs='+', type=int, default=[2], help='Attention layers.')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks.')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--multiplier', type=float, default=2., help='Multiplier.')
    parser.add_argument('--beta_1', type=float, default=1e-4, help='Beta 1.')
    parser.add_argument('--beta_T', type=float, default=0.02, help='Beta T.')
    parser.add_argument('--img_size', type=int, default=32, help='Image size.')
    parser.add_argument('--grad_clip', type=float, default=1., help='Gradient clipping.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use.')
    parser.add_argument('--training_load_weight', type=str, default=None, help='Training load weight.')
    parser.add_argument('--save_weight_dir', type=str, default='./Checkpoints/', help='Directory to save weights.')
    parser.add_argument('--test_load_weight', type=str, default='ckpt_199_.pt', help='Test load weight.')
    parser.add_argument('--sampled_dir', type=str, default='./SampledImgs/', help='Directory for sampled images.')
    parser.add_argument('--sampledNoisyImgName', type=str, default='NoisyNoGuidenceImgs.png', help='Sampled noisy image name.')
    parser.add_argument('--sampledImgName', type=str, default='SampledNoGuidenceImgs.png', help='Sampled image name.')
    parser.add_argument('--nrow', type=int, default=8, help='Number of rows for image grid.')

    args = parser.parse_args()
    main(args)