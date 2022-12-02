## This file contains schedulers that are not natively in torch.
# The default parameters have been tuned on an audio source separation task.
# More information are available in this report :
#
## If you like to add more schedulers:
# Subclass them from _LRScheduler so that you can easily check whether an object is a scheduler or not.
# Make sure that all the new schedulers are derived from torch's formalism.
# Some schedulers may include a stop criterion. In such case, use `self.stopping_required` in the scheduler.
#
## Don't hesitate to check torch's source code :
# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html

## Exemple of existing torch schedulers
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CASWR
# from torch.optim.lr_scheduler import ReduceLROnPlateau as ROP
# from torch.optim.lr_scheduler import StepLR

import warnings
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class ROP2(_LRScheduler):
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
    def __init__(self,
                 optimizer: Optimizer,

                 factor: float = 0.5,
                 patience: int = 3,
                 N_reduces: int = 5,

                 verbose: bool = False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.patience = patience
        self.N_reduces = N_reduces
        self.verbose = verbose
        self.num_bad_epochs = 0
        self.previous_val_loss = None
        self.last_epoch = 0
        self.num_bad_epochs = 0
        self.N_reduces_counter = 0
        self.stopping_required = False

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        self.found_bad_epoch = False

        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        # Safe case that should only happen at first epoch
        if self.previous_val_loss is None:
            self.previous_val_loss = current
            return

        if current >= self.previous_val_loss:
            self.found_bad_epoch = True
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self._reduce_lr(epoch=epoch)

        self.previous_val_loss = current

    def _reduce_lr(self, epoch):
        self.N_reduces_counter += 1

        if self.N_reduces_counter >= self.N_reduces:
            self.stopping_required = True

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))


class Chained_ROP2(_LRScheduler):
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
    def __init__(self,
                 optimizer: Optimizer,

                 factor: float = 0.5,
                 patience: int = 3,
                 N_reduces: int = 5,
                 N_rop: int = 3,

                 verbose: bool = False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.patience = patience
        self.N_reduces = N_reduces
        self.N_rop = N_rop
        self.verbose = verbose
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.num_bad_epochs = 0
        self.previous_val_loss = None
        self.last_epoch = 0
        self.num_bad_epochs = 0
        self.N_reduces_counter = 0
        self.N_rop_counter = 0
        self.stopping_required = False

    def _reset_ROP(self):
        self.N_rop_counter += 1
        if self.N_rop_counter >= self.N_rop:
            self.stopping_required = True
        self.optimizer.param_groups[0]['lr'] = self.initial_lr
        self.previous_val_loss = None
        self.num_bad_epochs = 0
        self.N_reduces_counter = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        self.found_bad_epoch = False

        self.last_epoch += 1

        # Safe case that should only happen at first epoch of a ROP
        if self.previous_val_loss is None:
            self.previous_val_loss = current
            return

        if current >= self.previous_val_loss:
            self.found_bad_epoch = True
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self._reduce_lr()

        self.previous_val_loss = current

    def _reduce_lr(self):
        self.N_reduces_counter += 1

        if self.N_reduces_counter >= self.N_reduces:
            self._reset_ROP()
            return

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(self.last_epoch, float) else
                             "%.5d") % self.last_epoch
                print('Epoch {}: reducing learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))


class CASWR_ROP2(_LRScheduler):
    """
    Creates a scheduler composed of a first phase of `N_annealings`
    loops of CASWR, then a ROP2 with stop criterion.
    """
    def __init__(self,
                 optimizer: Optimizer,

                 lr_min: float = 0,
                 T_0: int = 8,
                 N_annealings: int = 8,

                 factor: float = 0.5,
                 patience: int = 3,
                 N_reduces: int = 5,

                 verbose: bool = False,
                 T_mult: float = 1):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Other attributes
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self.N_annealings = N_annealings
        self.patience = patience
        self.N_reduces = N_reduces
        self.verbose = verbose
        self.last_epoch = 0
        self.phase = 'CASWR'
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.diff = (self.initial_lr - self.lr_min) / 2
        self.lr_schedule = (self.diff * (np.cos(t * np.pi / self.T_0) + 1)
                            for t in range(self.T_0))
        next(self.lr_schedule)
        self.annealing_counter = 0
        self.num_bad_epochs = 0
        self.previous_val_loss = None
        self.num_bad_epochs = 0
        self.N_reduces_counter = 0
        self.stopping_required = False

    def _switch_phase(self):
        self.phase = 'ROP2'
        self.optimizer.param_groups[0]['lr'] = self.initial_lr

    def _reset_annealing(self):
        self.lr_schedule = (self.diff * (np.cos(t * np.pi / self.T_0) + 1)
                            for t in range(self.T_0))
        self.optimizer.param_groups[0]['lr'] = next(self.lr_schedule)
        self.annealing_counter += 1

    def _adjust_lr(self):
        new_lr = next(self.lr_schedule)
        self.optimizer.param_groups[0]['lr'] = new_lr

        if self.verbose:
            epoch_str = ("%.2f" if isinstance(self.last_epoch, float) else
                         "%.5d") % self.last_epoch
            print('Epoch {}: adjusting learning rate'
                  ' to {:.4e}.'.format(epoch_str, new_lr))

    def _reduce_lr(self):
        self.N_reduces_counter += 1
        if self.N_reduces_counter >= self.N_reduces:
            self.stopping_required = True

        old_lr = float(self.optimizer.param_groups[0]['lr'])
        new_lr = old_lr * self.factor
        self.optimizer.param_groups[0]['lr'] = new_lr

        if self.verbose:
            epoch_str = ("%.2f" if isinstance(self.last_epoch, float) else
                         "%.5d") % self.last_epoch
            print('Epoch {}: reducing learning rate'
                  ' to {:.4e}.'.format(epoch_str, new_lr))

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        self.last_epoch += 1

        if self.phase == 'CASWR':
            try:
                self._adjust_lr()
            except StopIteration:
                self._reset_annealing()

            if self.annealing_counter >= self.N_annealings:
                self._switch_phase()

        elif self.phase == 'ROP2':
            self.found_bad_epoch = False

            # Safe case that only happens at first epoch of ROP2
            if self.previous_val_loss is None:
                self.previous_val_loss = current
                return

            if current >= self.previous_val_loss:
                self.found_bad_epoch = True
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                self._reduce_lr()
                self.num_bad_epochs = 0

            self.previous_val_loss = current


class ROP2_CASWR_Alt(_LRScheduler):
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

    def __init__(self,
                 optimizer: Optimizer,
                 starting_phase: str = 'ROP2',

                 factor: float = 0.5,
                 patience: int = 3,
                 N_reduces: int = 5,

                 lr_min: float = 0,
                 T_0: int = 8,

                 verbose: bool = False):
        # Errors
        if starting_phase not in ['CASWR', 'ROP2']:
            raise Exception('Starting phase is incorrect, it should be "CASWR" or "ROP2"')
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Attributes
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.patience = patience
        self.N_reduces = N_reduces
        self.lr_min = lr_min
        self.T_0 = T_0
        self.num_bad_epochs = 0
        self.phase = starting_phase
        self.verbose = verbose
        self.counter = 0
        self.last_epoch = 0
        self.lr_schedule = None
        self.previous_val_loss = None

    def _switch_phase(self):
        if self.phase == 'CASWR':
            print('Switching to ROP2 phase')
            self.phase = 'ROP2'
            self.optimizer.param_groups[0]['lr'] = self.initial_lr
            self.counter = 0
            self.previous_val_loss = None

        elif self.phase == 'ROP2':
            print('Switching to CASWR phase')
            self.phase = 'CASWR'
            self.found_bad_epoch = False
            diff = (self.initial_lr - self.lr_min) / 2
            self.lr_schedule = (diff * (np.cos(t * np.pi / self.T_0) + 1)
                                for t in range(self.T_0))

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr

            if self.verbose:
                epoch_str = ("%.2f" if isinstance(self.last_epoch, float) else
                             "%.5d") % self.last_epoch
                print('Epoch {}: reducing learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    def _adjust_lr(self):
        new_lr = next(self.lr_schedule)
        self.optimizer.param_groups[0]['lr'] = new_lr

        if self.verbose:
            epoch_str = ("%.2f" if isinstance(self.last_epoch, float) else
                         "%.5d") % self.last_epoch
            print('Epoch {}: adjusting learning rate'
                  ' to {:.4e}.'.format(epoch_str, new_lr))

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        self.last_epoch += 1

        if self.phase == 'CASWR':
            try:
                self._adjust_lr()
            except StopIteration:
                self._switch_phase()

        elif self.phase == 'ROP2':
            self.found_bad_epoch = False

            # Safe case that only happens at first epoch of ROP2
            if self.previous_val_loss is None:
                self.previous_val_loss = current
                return

            if current >= self.previous_val_loss:
                self.found_bad_epoch = True
                self.num_bad_epochs += 1

            if self.num_bad_epochs == self.patience:
                self._reduce_lr()
                self.num_bad_epochs = 0
                self.counter += 1

            if self.counter == self.N_reduces:
                self._switch_phase()

            self.previous_val_loss = current


class ROCP(_LRScheduler):
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
    def __init__(self,
                 optimizer: Optimizer,

                 factor: float = 0.5,
                 threshold: float = .005,
                 N_reduces: int = 5,

                 verbose: bool = False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.threshold = threshold
        self.N_reduces = N_reduces
        self.verbose = verbose
        self.last_epoch = 0
        self.prev_val = None
        self.N_reduces_counter = 0
        self.stopping_required = False

    def step(self, metrics, epoch=None):
        current = float(metrics)

        # Safe case that only happens at first epoch
        if self.prev_val is None:
            self.prev_val = current
            return

        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if abs(current-self.prev_val)/current < self.threshold:
            self._reduce_lr(epoch=epoch)

        self.prev_val = current

    def _reduce_lr(self, epoch):
        self.N_reduces_counter += 1
        self.reduce_required = False

        if self.N_reduces_counter >= self.N_reduces:
            self.stopping_required = True

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))
                print('Reduction {}/{}'.format(self.N_reduces_counter, self.N_reduces))