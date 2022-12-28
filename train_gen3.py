import argparse
import importlib
from utils import *


MODEL_DIR=None
DATA_DIR = 'data/'
OBJ_DIR=None
CORE_DIR = None
PROJECT='gen3'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='mini_imagenet',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100) # original version
    parser.add_argument('-epochs_new', type=int, default=60)
    parser.add_argument('-lr_base', type=float, default=0.05)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-lr_new_enc', type=float, default=0.03)
    parser.add_argument('-schedule', type=str, default='Milestone',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-schedule_new', type=str, default='Milestone',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 80])  # original
    parser.add_argument('-milestones_new', nargs='+', type=int, default=[20, 50])  # original
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-step_new', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=64)
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos',
                                 'avg_cos'])  # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    #parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-lrg', type=float, default=0.0002)  # lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=0)
    #parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    #parser.add_argument('-gpu', default='0,1')
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    # parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-debug', action='store_true')

    parser.add_argument("-use_selfsup_fe", default=False)
    parser.add_argument("-freeze_backbone", default=False)
    parser.add_argument('-selfsup_arch', type=str, default='resnet18',
                        help='resnet18, resnet50')
    #parser.add_argument('-selfsup_arch_num_features', type=int, default=512,
    #                    help='self-sup-resnet50:2048, self-sup-resnet18: 512')
    parser.add_argument('-selfsup_alg', type=str, default='simclr',
                        choices='simclr, mocov2')
    parser.add_argument('-log_path', type=str, default='results')
    #parser.add_argument('-seed', type=int, default=1, help='if 0: random. otherwise set seed.')
    parser.add_argument('-seed', type=int, default=10, help='if 0: random. otherwise set seed.')
    parser.add_argument('-varsess', default=True , help='session order changing')

    parser.add_argument('-shot', type=int, default=5,
    #parser.add_argument('-shot', type=int, default=20,
                        help='for varsess==False, only 5 should be used since data fixed in txt files ')

    parser.add_argument('-base_class', type=int, default=60,
                        help='for varsess==False, None results default setting',
                        choices='80,90,100,150,190,None')
    parser.add_argument('-way', type=int, default=5,
                        help='for varsess==False, None results default setting',
                        choices='5,10,None')

    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new'
                             'if varsess==F, we use txt so always use 0'
                             'Otherwise, use 0 or wanted value. Especially if shot is big'
                             'if way/shot is big, using 0 may occur CUDA memory error, '
                             'so set appropriate value for this e.g. 16')

    #parser.add_argument('-num_sampled', type=int, default=100)
    parser.add_argument('-num_sampled', type=int, default=0)
    parser.add_argument('-num_sampled_n', type=int, default=15)
    #parser.add_argument('-num_sampled', type=int, default=30)
    parser.add_argument('-tukey_beta', type=float, default=0.5)

    parser.add_argument('-m', type=float, default=0.0)
    parser.add_argument('-s', type=float, default=16.0)
    parser.add_argument('-angle_mode', default='cosface', choices=[None,'arcface','cosface'])
    parser.add_argument('-obj_root', type=str, default=OBJ_DIR)
    parser.add_argument('-core_root', type=str, default=CORE_DIR)

    parser.add_argument('-w_ce', type=float, default=1.0)
    parser.add_argument('-w_prev_ce', type=float, default=0.0)
    parser.add_argument('-w_ce5', type=float, default=1.0)


    parser.add_argument('-w_l2', type=float, default=5.0)
    parser.add_argument('-w_l22', type=float, default=5.0)

    parser.add_argument('-w_distill4', type=float, default=400.0)

    parser.add_argument('-m_atcl_n', type=float, default=0.1)
    parser.add_argument('-w_atcl_n', type=float, default=0.1)

    return parser


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
