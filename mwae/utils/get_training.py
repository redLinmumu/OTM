import torch


def get_optimizer(parameters, type, lr, **kwargs):
    """
        An instantiated optimizer
    """

    if isinstance(type, str):
        if hasattr(torch.optim, type):
            # if isinstance(parameters, torch.nn.Module):
            #     return getattr(torch.optim, type)(parameters.parameters(), lr,
            #                                       **tmp_kwargs)
            # else:
            return getattr(torch.optim, type)(parameters, lr, **kwargs)
        else:
            raise NotImplementedError(
                'Optimizer {} not implement'.format(type))
    else:
        raise TypeError()


def get_scheduler(optimizer, type, **kwargs):
    """
    This function builds an instance of scheduler.

    Args:
        optimizer: optimizer to be scheduled
        type: type of scheduler
        **kwargs: kwargs dict

    Returns:
        An instantiated scheduler.

    Note:
        Please follow ``contrib.scheduler.example`` to implement your own \
        scheduler.
    """
    if isinstance(type, str):
        if hasattr(torch.optim.lr_scheduler, type):
            return getattr(torch.optim.lr_scheduler, type)(optimizer, **kwargs)
        else:
            raise NotImplementedError(
                'Scheduler {} not implement'.format(type))
    else:
        raise TypeError()