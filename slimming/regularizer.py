import re

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from slimming.utils import ModifierYAML


def gather_bn_weights(model):
    weight_list = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            weight_list.append(m.weight.data.abs().clone())
    all_bn_weights = torch.cat(weight_list)
    return all_bn_weights


def plot_histogram(model, writer, epoch):
    all_bn_weights = gather_bn_weights(model)
    writer.add_histogram('BN_weights', all_bn_weights.cpu().numpy(), epoch)
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # weight_list.append(m.weight.data.abs().clone())
            if len(m.weight.data.abs().clone().cpu().numpy()) == 0:
                continue
            writer.add_histogram('BN_weights/' + name, m.weight.data.abs().clone().cpu().numpy(), epoch + 1)


def get_bn_layers(model, ignores):
    bn_modules = []
    for name, m in model.named_modules():
        # if isinstance(m, nn.BatchNorm2d) and name not in ignores:
        #     bn_modules.append(m)
        if isinstance(m, nn.BatchNorm2d):
            choice = True
            for ignore in ignores:
                if re.match(ignore, name):
                    choice = False
                    break
            if choice:
                bn_modules.append(m)

    return bn_modules


@ModifierYAML()
class BaseReg(object):
    def __init__(self, lbd=0.0002, ignores=None, start_epoch=0, end_epoch=100, **kwargs):

        self.lbd = lbd
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        if ignores is None:
            ignores = []
        self.ignores = ignores

        self.model = None
        self.modules = None

    def set_model(self, model):
        pass

    def sparsity(self, **kwargs):
        pass

    def log_update(self, logger, epoch, **kwargs):
        if isinstance(logger, SummaryWriter):
            logger.add_scalar('train/reg_loss', self.sparsity(), epoch)
            plot_histogram(self.model, logger, epoch)

    def loss_update(self, loss, **kwargs):
        """
        Modify the loss to include the norms for the outputs of the layers
        being modified.

        :param loss: The calculated loss tensor
        :return: the modified loss tensor
        """
        pass

    def optimizer_pre_step(self, **kwargs):
        pass

    def optimizer_post_step(self, **kwargs):
        pass

    def yaml_key(self):
        return "!{}".format(self.__class__.__name__)


@ModifierYAML()
class PolarizationReg(BaseReg):
    def __init__(self, t=1., upper_bound=1., **kwargs):
        super().__init__(**kwargs)

        self.t = t
        self.upper_bound = upper_bound

    def set_model(self, model):
        self.model = model
        self.modules = get_bn_layers(model, ignores=self.ignores)

    def clamp(self):
        for m in self.modules:
            # m.weight.data.clamp_(lower_bound, upper_bound)
            m.weight.data.clamp_(0., self.upper_bound)

    def sparsity(self, ):
        # compute global mean of all sparse vectors
        len_modules = sum(map(lambda module: module.weight.data.shape[0], self.modules))
        sparse_weights_mean = torch.sum(torch.stack(
            list(map(lambda module: torch.sum(module.weight), self.modules)))) / len_modules

        sparsity_term = 0.
        for m in self.modules:
            sparsity_term += self.t * torch.sum(torch.abs(m.weight)) \
                             - torch.sum(torch.abs(m.weight - sparse_weights_mean))

        return sparsity_term

    def loss_update(self, loss, **kwargs):
        sparsity_term = self.sparsity() * self.lbd
        loss += sparsity_term

        return loss

    def optimizer_pre_step(self):
        self.clamp()


@ModifierYAML()
class L1Reg(BaseReg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_model(self, model):
        self.model = model
        self.modules = get_bn_layers(model, ignores=self.ignores)

    def sparsity(self, ):
        sparsity_term = 0.
        for m in self.modules:
            sparsity_term += torch.sum(torch.abs(m.weight))

        return sparsity_term

    def loss_update(self, loss, **kwargs):
        sparsity_term = self.sparsity() * self.lbd
        loss += sparsity_term

        return loss


@ModifierYAML()
class NoneReg(BaseReg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_model(self, model):
        self.model = model
        self.modules = get_bn_layers(model, ignores=self.ignores)

    def sparsity(self, ):
        sparsity_term = 0.
        for m in self.modules:
            sparsity_term += torch.sum(torch.abs(m.weight))

        return sparsity_term

    def loss_update(self, loss, **kwargs):
        return loss
