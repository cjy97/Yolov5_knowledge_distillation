from torch.optim.optimizer import Optimizer

from slimming.regularizer import PolarizationReg, L1Reg, NoneReg

__all__ = ["ScheduledOptimizer"]


class ScheduledOptimizer(Optimizer):
    """
    An optimizer wrapper to handle applying modifiers according to their schedule
    to both the passed in optimizer and the module.
    """

    def __init__(self, regularzier, optimizer=None, model=None, logger=None):
        self._optimizer = optimizer
        self._module = model
        self._logger = logger
        self._regularzier = regularzier

        # self._regularzier = PolarizationReg(model=module)
        # self._regularzier = L1Reg(model=module)
        # self._regularzier = NoneReg(model=module)

        self.cur_epoch = -1

    # def loss_update(self, loss):
    #     return self._regularzier.loss_update(loss)

    def update(self, loss, epoch):
        new_loss = self._regularzier.loss_update(loss)
        if epoch > self.cur_epoch:
            # self._regularzier.log_update(self._logger, epoch)
            self.cur_epoch = epoch
        return new_loss

    # def log_update(self, epoch):
    #     if epoch > self.cur_epoch:
    #         self._regularzier.log_update(self._logger, epoch)
    #         self.cur_epoch = epoch

    def optimizer_pre_step(self):
        self._regularzier.optimizer_pre_step()

    def optimizer_post_step(self):
        self._regularzier.optimizer_post_step()

    def step(self, closure=None):
        """
        Called to perform a step on the optimizer activation normal.
        Updates the current epoch based on the step count.
        Calls into modifiers before the step happens.
        Calls into modifiers after the step happens.

        :param closure: optional closure passed into the contained optimizer
            for the step
        """
        self.optimizer_pre_step()
        self._optimizer.step(closure)
        self.optimizer_post_step()

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value

    def add_param_group(self, param_group):
        self._optimizer.add_param_group(param_group)

    def state_dict(self):
        return (self._optimizer.state_dict(),)

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self._optimizer.zero_grad()
