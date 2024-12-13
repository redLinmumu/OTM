import copy
import logging
import os
from pathlib import Path
from utils.yacs_config import CfgNode, _merge_a_into_b, Argument

logger = logging.getLogger(__name__)


def set_help_info(cn_node, help_info_dict, prefix=""):
    for k, v in cn_node.items():
        if isinstance(v, Argument) and k not in help_info_dict:
            help_info_dict[prefix + k] = v.description
        elif isinstance(v, CN):
            set_help_info(v,
                          help_info_dict,
                          prefix=f"{k}." if prefix == "" else f"{prefix}{k}.")


class CN(CfgNode):
    """
    An extended configuration system based on [yacs]( \
    https://github.com/rbgirshick/yacs). \
    The two-level tree structure consists of several internal dict-like \
    containers to allow simple key-value access and management.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        init_dict = super(CN, self).__init__(init_dict, key_list, new_allowed)
        self.__cfg_check_funcs__ = list()  # to check the config values
        # validity
        self.__help_info__ = dict()  # build the help dict

        self.is_ready_for_run = False  # whether this CfgNode has checked its
        # validity, completeness and clean some un-useful info

        if init_dict:
            for k, v in init_dict.items():
                if isinstance(v, Argument):
                    self.__help_info__[k] = v.description
                elif isinstance(v, CN) and "help_info" in v:
                    for name, des in v.__help_info__.items():
                        self.__help_info__[name] = des

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(name)

    def clear_aux_info(self):
        """
        Clears all the auxiliary information of the CN object.
        """
        if hasattr(self, "__cfg_check_funcs__"):
            delattr(self, "__cfg_check_funcs__")
        if hasattr(self, "__help_info__"):
            delattr(self, "__help_info__")
        if hasattr(self, "is_ready_for_run"):
            delattr(self, "is_ready_for_run")
        for v in self.values():
            if isinstance(v, CN):
                v.clear_aux_info()

    def print_help(self, arg_name=""):
        """
        print help info for a specific given ``arg_name`` or \
        for all arguments if not given ``arg_name``

        Args:
            arg_name: name of specific args
        """
        if arg_name != "" and arg_name in self.__help_info__:
            print(f"  --{arg_name} \t {self.__help_info__[arg_name]}")
        else:
            for k, v in self.__help_info__.items():
                print(f"  --{k} \t {v}")

    def register_cfg_check_fun(self, cfg_check_fun):
        """
        Register a function that checks the configuration node.

        Args:
            cfg_check_fun: function for validation the correctness of cfg.
        """
        self.__cfg_check_funcs__.append(cfg_check_fun)

    def merge_from_file(self, cfg_filename, check_cfg=True):
        """
        load configs from a yaml file, another cfg instance or a list \
        stores the keys and values.

        Args:
            cfg_filename: file name of yaml file
            check_cfg: whether enable ``assert_cfg()``
        """
        # cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        with open(cfg_filename, "r") as f:
            cfg = self.load_cfg(f)
        cfg.clear_aux_info()
        self.merge_from_other_cfg(cfg)
        # self.__cfg_check_funcs__.clear()
        # self.__cfg_check_funcs__.extend(cfg_check_funcs)
        # self.assert_cfg(check_cfg)
        # set_help_info(self, self.__help_info__)

    def merge_from_other_cfg(self, cfg_other, check_cfg=True):
        """
        load configs from another cfg instance

        Args:
            cfg_other: other cfg to be merged
            check_cfg: whether enable ``assert_cfg()``
        """
        # cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        _merge_a_into_b(cfg_other, self, self, [])
        # self.__cfg_check_funcs__.clear()
        # self.__cfg_check_funcs__.extend(cfg_check_funcs)
        # self.assert_cfg(check_cfg)
        # set_help_info(self, self.__help_info__)

    def merge_from_list(self, cfg_list, check_cfg=True):
        """
        load configs from a list stores the keys and values. \
        modified ``merge_from_list`` in ``yacs.config.py`` to allow adding \
        new keys if ``is_new_allowed()`` returns True \

        Args:
            cfg_list: list of pairs of cfg name and value
            check_cfg: whether enable ``assert_cfg()``
        """
        if hasattr(self, '__cfg_check_funcs__'):
            self.__cfg_check_funcs__.clear()
        else:
            self.__cfg_check_funcs__ = []
        # cfg_check_funcs = copy.copy(self.__cfg_check_funcs__)
        super().merge_from_list(cfg_list)
        # self.__cfg_check_funcs__.extend(cfg_check_funcs)
        # self.assert_cfg(check_cfg)
        # if not hasattr(self, '__help_info__'):
        #     self.__help_info__ = dict()
        # set_help_info(self, self.__help_info__)

    def assert_cfg(self, check_cfg=True):
        """
        check the validness of the configuration instance

        Args:
            check_cfg: whether enable checks
        """
        if check_cfg:
            for check_func in self.__cfg_check_funcs__:
                check_func(self)

    def clean_unused_sub_cfgs(self):
        """
        Clean the un-used secondary-level CfgNode, whose ``.use`` \
        attribute is ``True``
        """
        for v in self.values():
            if isinstance(v, CfgNode) or isinstance(v, CN):
                # sub-config
                if hasattr(v, "use") and v.use is False:
                    for k in copy.deepcopy(v).keys():
                        # delete the un-used attributes
                        if k == "use":
                            continue
                        else:
                            del v[k]

    def check_required_args(self):
        """
        Check required arguments.
        """
        for k, v in self.items():
            if isinstance(v, CN):
                v.check_required_args()
            if isinstance(v, Argument) and v.required and v.value is None:
                logger.warning(f"You have not set the required argument {k}")

    def de_arguments(self):
        """
        some config values are managed via ``Argument`` class, this function \
        is used to make these values clean without the ``Argument`` class, \
        such that the potential type-specific methods work correctly, \
        e.g., ``len(cfg.federate.method)`` for a string config
        """
        for k, v in copy.deepcopy(self).items():
            if isinstance(v, CN):
                self[k].de_arguments()
            if isinstance(v, Argument):
                self[k] = v.value

    def ready_for_run(self, check_cfg=True):
        """
        Check and cleans up the internal state of cfg and save cfg.

        Args:
            check_cfg: whether enable ``assert_cfg()``
        """
        self.assert_cfg(check_cfg)
        self.clean_unused_sub_cfgs()
        self.check_required_args()
        self.de_arguments()
        self.is_ready_for_run = True

    def freeze(self, inform=True, save=True, check_cfg=True):
        """
        (1) make the cfg attributes immutable;
        (2) if ``save==True``, save the frozen cfg_check_funcs into \
            ``self.outdir/config.yaml`` for better reproducibility;
        (3) if ``self.wandb.use==True``, update the frozen config
        """
        self.ready_for_run(check_cfg)
        super(CN, self).freeze()

        if save:  # save the final cfg
            Path(self.outdir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.outdir, "config.yaml"),
                      'w') as outfile:
                from contextlib import redirect_stdout
                with redirect_stdout(outfile):
                    tmp_cfg = copy.deepcopy(self)
                    tmp_cfg.clear_aux_info()
                    print(tmp_cfg.dump())

            if inform:
                logger.info("the used configs are: \n" + str(tmp_cfg))



def init_global_cfg(cfg):
    """
    This function sets the default config value.

    (1) Note that for an experiment, only part of the arguments will be used \
    The remaining unused arguments won't affect anything. \
    So feel free to register any argument in graphgym.contrib.config
    (2) We support more than one levels of configs, e.g., cfg.dataset.name
    """

    # ---------------------------------------------------------------------- #
    # Basic options, first level configs
    # ---------------------------------------------------------------------- #

    cfg.backend = 'torch'

    # Whether to print verbose logging info
    cfg.verbose = 1

    # How many decimal places we print out using logger
    cfg.print_decimal_digits = 20

    # Random seed
    cfg.seed = 0

    # Whether to use GPU
    cfg.use_gpu = False

    # Specify the device
    cfg.device = -1

    # Path of configuration file
    cfg.cfg_file = ''

    cfg.baseline = 'mwae'

    cfg.outdir = CN()

    # The dir used to save log, exp_config, models, etc,.
    cfg.outdir.dir = 'exp'
    cfg.outdir.expname = ''  # detailed exp name to distinguish different sub-exp
    cfg.outdir.expname_tag = ''  # detailed exp tag to distinguish different
    # sub-exp with the same expname
    cfg.outdir.save_to = 'save'
    cfg.outdir.restore_from = 'save'

    cfg.split = CN()
    cfg.split.tau = 1.0
    cfg.split.eta = 1.0
    cfg.split.alpha = 1.0

    cfg.data = CN()
    cfg.data.type = '3s'
    cfg.data.unaligned_rate = 0.0
    cfg.data.splits = [0.7, 0.2, 0.1]
    cfg.data.correspondence = True
    cfg.data.root = 'data/'
    cfg.data.cluster_num = 1
    cfg.data.num_views = 1
    cfg.data.modality_feature_names = []
    cfg.data.modality_feature_dims = []
    cfg.data.train_samples_num = 0
    cfg.data.valid_samples_num = 0
    cfg.data.test_samples_num = 0
    cfg.data.raw_tsne = False 
    cfg.data.valid_tsne = False 
    cfg.data.is_filter = False # for cal7
    cfg.data.filter_num = 2 # for cal7

    cfg.model = CN()
    cfg.model.h_dim = 10
    cfg.model.z_dim = 10
    cfg.model.is_save = False
    cfg.model.is_load = False

    cfg.mixer = CN()
    cfg.mixer.inner_iter = 5
    cfg.mixer.loss_fn = 'L2'  # 'KL'
    cfg.mixer.gw_method = 'gw'
    cfg.mixer.consist = False
    cfg.mixer.gamma = 1e-2
    cfg.mixer.f_alpha = 0.5
    cfg.mixer.fuse = 'add' # 'con','bary'

    cfg.train = CN()
    cfg.train.batch_or_epoch = 'epoch'
    cfg.train.local_update_steps = 10
    cfg.train.batch_size = 10
    
    cfg.train.early_stop = True
    cfg.train.patience = 5
    
    cfg.train.optimizer = CN()
    cfg.train.optimizer.type = 'Adam'
    cfg.train.optimizer.lr = 1e-3
    cfg.train.optimizer.weight_decay = 0.0

    cfg.train.scheduler = CN()
    cfg.train.scheduler.type = 'ReduceLROnPlateau'
    cfg.train.scheduler.mode = 'max'
    cfg.train.scheduler.factor = 0.9
    cfg.train.scheduler.patience = 20
    cfg.train.scheduler.threshold = 1e-7
    cfg.train.scheduler.min_lr = 1e-10

    cfg.nni = False
    cfg.metric = 0   # 0:acc, 1:nmi, -1:ami

# Global config object
global_cfg = CN()
init_global_cfg(global_cfg)
global_cfg.clear_aux_info()

