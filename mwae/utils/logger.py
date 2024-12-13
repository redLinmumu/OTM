import logging
import os
import time
from datetime import datetime
import numpy as np
import re
class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629"""
    def __init__(self, fmt):
        super().__init__()
        grey = '\x1b[38;21m'
        blue = '\x1b[38;5;39m'
        yellow = "\x1b[33;20m"
        red = '\x1b[38;5;196m'
        bold_red = '\x1b[31;1m'
        reset = '\x1b[0m'

        self.FORMATS = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: blue + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)



class LoggerPrecisionFilter(logging.Filter):
    def __init__(self, precision):
        super().__init__()
        self.print_precision = precision

    def str_round(self, match_res):
        return str(round(eval(match_res.group()), self.print_precision))

    def filter(self, record):
        # use regex to find float numbers and round them to specified precision
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)
        if record.msg != "":
            if re.search(r"([-+]?\d+\.\d+)", record.msg):
                record.msg = re.sub(r"([-+]?\d+\.\d+)", self.str_round,
                                    record.msg)
        return True


def update_logger(cfg, clear_before_add=False):
    root_logger = logging.getLogger("mwae")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        handler.setFormatter(CustomFormatter(fmt))

        root_logger.addHandler(handler)

    # update level
    if cfg.verbose > 0:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN
        root_logger.warning("Skip DEBUG/INFO messages")
    root_logger.setLevel(logging_level)

    # ================ create outdir to save log, exp_config, models, etc,.
    outdir = cfg.outdir.dir
    expname = cfg.outdir.expname
    expname_tag = cfg.outdir.expname_tag
    if outdir == "":
        outdir = os.path.join(os.getcwd(), "exp")
    if expname == "":
        expname = f"{cfg.baseline}_{cfg.model.type}_on" \
                      f"_{cfg.data.type}_lr{cfg.train.optimizer.lr}_lste" \
                      f"p{cfg.train.local_update_steps}"
    if expname_tag:
        expname = f"{expname}_{expname_tag}"
    outdir = os.path.join(outdir, expname)

    # if exist, make directory with given name and time
    if os.path.isdir(outdir) and os.path.exists(outdir):
        outdir = os.path.join(outdir, "sub_exp" +
                              datetime.now().strftime('_%Y%m%d%H%M%S')
                              )  # e.g., sub_exp_20220411030524
        while os.path.exists(outdir):
            time.sleep(1)
            outdir = os.path.join(
                outdir,
                "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
        cfg.outdir.dir = outdir
    # if not, make directory with given name
    os.makedirs(outdir, exist_ok=True)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    root_logger.addHandler(fh)

    # set print precision for terse logging
    np.set_printoptions(precision=cfg.print_decimal_digits)
    precision_filter = LoggerPrecisionFilter(cfg.print_decimal_digits)
    # attach the filter to the fh handler to propagate the filter, since
    # "Filters, unlike levels and handlers, do not propagate",
    # ref https://stackoverflow.com/questions/6850798/why-doesnt-filter-
    # attached-to-the-root-logger-propagate-to-descendant-loggers
    for handler in root_logger.handlers:
        handler.addFilter(precision_filter)

    import socket
    root_logger.info(f"the current machine is at"
                     f" {socket.gethostbyname(socket.gethostname())}")
    root_logger.info(f"the current dir is {os.getcwd()}")
    root_logger.info(f"the output dir is {cfg.outdir.dir}")
