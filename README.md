# About

**torch_custom_lr_schedulers** contains several torch custom learning rate schedulers.

The default parameters have been tuned on a speech enhancement task.

Please find more experimental details about [ROP2](https://github.com/SteelSeries/torch_custom_lr_schedulers/edit/main/README.md#class-rop2_lrscheduler), [Chained_ROP2](https://github.com/SteelSeries/torch_custom_lr_schedulers/edit/main/README.md#class-chained_rop2_lrscheduler), [CASWR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
) and [CASWR_ROP2](https://github.com/SteelSeries/torch_custom_lr_schedulers/edit/main/README.md#class-caswr_rop2_lrscheduler
) in the [attached report](Learning_rate_scheduling_and_gradient_clipping_for_audio_source_separation.pdf). 

This work has been carried during Matéo Vial internship at [SteelSeries](https://steelseries.com) France.

# Main content

### Class ROP2(_LRScheduler):
    """
    Implements a scheduler similar to the classic "ReduceLROnPlateau", except that
    this scheduler only requires a `patience` number of epochs for which the loss
    is less satisfying than the loss of the epoch directly before (This scheduler
    does not keep memory of the best epoch, unlike standard ReduceLROnPlateau).
    This has the advantage of lowering the loss faster.

    The code is greatly inspired from torch.optim.lr_scheduler.ReduceLROnPlateau
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau

    Args:
        factor: float
            multiplicative factor applied to lr when a plateau is found
        patience: int
            number of epochs which loss is worse than the previous epochs
            needed to consider a plateau
        N_reduces: int
            Training stops after the lr has been reduced `N_reduces` times.
            Acts as a stopping criterion.
        verbose: bool
            if True, will print info on the console when lr is reduced

    """
### Class Chained_ROP2(_LRScheduler):
    """
    Implements a scheduler made of several ROP2 with similar parameters
    chained together

    Args:
        factor: float
            multiplicative factor applied to lr when a plateau is found
        patience: int
            number of epochs which loss is worse than the previous epochs
            needed to consider a plateau
        N_reduces: int
            current ROP stops after the lr has been reduced `N_reduces` times.
            Then the next ROP starts.
        N_ROP: int
            Number of ROPS before stopping
        verbose: bool
            if True, will print info on the console when lr is reduced

    """
### Class CASWR_ROP2(_LRScheduler):
    """
    Creates a scheduler composed of a first phase of `N_annealings`
    loops of CASWR, then a ROP2 with stop criterion.
    """

### Class ROP2_CASWR_Alt(_LRScheduler):
    """
    Scheduler which alternates a phase of ROP2 and a phase of CASWR.

    The ROP2 phase is a standard ROP2, taking as args `factor` and `patience`.
    An additional `N_reduces` arg is required to determine how many times the
    learning rate has to be reduced before it switches to the CASWR phase.

    The CASWR phase is a standard CASWR, taking as args `lr_min` and `T_0`.
    T_mult is not yet implemented and may not be.

    This scheduler takes an additional arg, `starting_phase` which should be
    equal to 'CASWR' or 'ROP2' depending on which phase should be the first one.
    """

### Class ROCP(_LRScheduler):
    """
    This is a variant of ROP2. ROCP stands for "Reduce on Clip-value Plateau".
    The criterion for reducing the learning rate is when the clip value doesn't vary
    too much from an epoch to another (see `threshold`).

    Args:
        factor: float
            multiplicative factor applied to lr when a plateau is found
        threshold: float
            Threshold used to determine when the lr should be reduced
        N_reduces: int
            Training stops after the lr has been reduced `N_reduces` times.
            Acts as a stopping criterion.
        verbose: bool
            if True, will print info on the console when lr is reduced

    """

## Reference

If you use **torch_custom_lr_schedulers** in your work, please cite:

```BibTeX
@article{
}
```

## License

The code of **torch_custom_lr_schedulers** is [-licensed](LICENSE).

## Disclaimer

If you plan to use **torch_custom_lr_schedulers** on copyrighted material, make sure you get proper authorization from right owners beforehand.
