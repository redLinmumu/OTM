import argparse
import sys



def parse_args(args=None):
    parser = argparse.ArgumentParser(description='mwae',
                                     add_help=False)
    # config file path -- robotics
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Config file path',
                        required=False,
                        type=str)
    
    # weight of mixer loss
    parser.add_argument('--beta',
                        default=1.0,
                        help='Config beta',
                        required=False,
                        type=float)
    # GPU device
    parser.add_argument('--device', type=int, default=0, required=False, help='GPU device index (non-negative integer)')

    # Early stop tolerance
    parser.add_argument('--tolerance', type=int, default=7, required=False, help='Tolerance of early stop (integer greater than 0)')


    parser.add_argument('--epoch', type=int, default=100, required=False, help='Number of epochs (integer greater than or equal to 0)')

    parser.add_argument("--origin",  default=True, action="store_false", help="not to run origin(default: True)")
    
    parser.add_argument("--mixer",  default=True,  action="store_false", help="not to run origin(default: True)")
    
    parser.add_argument('--noise', type=int, default=1, required=False, help='Level of noise: 0-10')

    
    parse_res = parser.parse_args(args)

    return parse_res

