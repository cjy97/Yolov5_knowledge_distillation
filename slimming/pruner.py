import copy
from datetime import datetime
from functools import reduce
from random import sample
import re
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from models.common import Bottleneck
from slimming.dependency import DependencyGraph
from slimming.pruning import prune_batchnorm, prune_conv
from slimming.utils import ModifierYAML


class Pruner(metaclass=ABCMeta):
    def __init__(self, choices=None, ignores=None, **kwargs):
        self.model = None
        self.candidate = []

        if choices:
            self.choices = choices
        else:
            self.choices = []

        if ignores:
            self.ignores = ignores
        else:
            self.ignores = []

    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def compress(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, **kwargs):
        pass

    def yaml_key(self):
        return "!{}".format(self.__class__.__name__)


@ModifierYAML()
class LayerPruner(Pruner):

    def __init__(self, num_layers=9, threshold=None, sparsity=None, random=False, bins=100, upper_bound=1.0, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.threshold = None
        self.sparsity = None
        if threshold:
            self.threshold = threshold
        elif sparsity:
            self.sparsity = sparsity

        self.random = random

        self.bins = bins
        self.upper_bound = upper_bound

        self.model = None
        self.pruning_unit = Bottleneck
        self.candidate = []

        self.selected_candidate = []
        self.layer_importance = []

    def set_model(self, model):
        self.model = model

        for name, m in model.named_modules():
            if isinstance(m, self.pruning_unit):
                # weight_list.append(m.weight.data.abs().clone())
                self.candidate.append([name, m])

        if self.sparsity:
            weight_list = []
            for name, m in self.candidate:
                for module in m.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        weight_list.append(module.weight.data.abs().clone())

            all_bn_weights = torch.cat(weight_list)
            k = int(all_bn_weights.shape[0] * self.sparsity)
            global_threshold = torch.topk(
                all_bn_weights.view(-1), k, largest=False)[0].max()
            self.threshold = global_threshold

        self.calc_importance()

    def calc_sparsity(self, bn):
        assert isinstance(bn, nn.BatchNorm2d)

        weight = bn.weight.data.clone()
        w_abs = weight.abs()

        if self.threshold:
            # if threshold > 0.2:
            #     print(f"WARNING: threshold might be too large: {threshold}")
            pruned = w_abs[w_abs < self.threshold]
            sparsity = len(pruned) / len(w_abs)
            return sparsity

        hist_y, hist_x = np.histogram(w_abs.cpu(), bins=self.bins, range=(0, self.upper_bound))
        hist_y_diff = np.diff(hist_y)
        for i in range(len(hist_y_diff) - 1):
            if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
                threshold = hist_x[i + 1]

                pruned = w_abs[w_abs < threshold]
                sparsity = len(pruned) / len(w_abs)
                return sparsity

    def calc_importance(self, verbose=True):

        for name, m in self.candidate:
            list_sparsity = []
            for module in m.modules():
                if isinstance(module, nn.BatchNorm2d):
                    list_sparsity.append(self.calc_sparsity(module))

            # prod_sparsity = reduce(lambda x, y: x * y, list_sparsity)
            # self.layer_importance.append([name, 1 - prod_sparsity])
            prod_sparsity = reduce(lambda x, y: (1 - x) * (1 - y), list_sparsity)
            self.layer_importance.append([name, prod_sparsity])
            if verbose:
                print(f"name: {name} importance: {prod_sparsity}")

        self.layer_importance.sort(key=lambda x: x[1])

    def compress(self, reverse=False, save_dir=None):
        num_layers = self.num_layers
        if self.random:
            self.selected_candidate = sample(self.layer_importance, num_layers)
        else:
            if reverse:
                self.selected_candidate = self.layer_importance[-num_layers:]
            else:
                self.selected_candidate = self.layer_importance[:num_layers]

        if save_dir:
            with open(save_dir / 'layer_importance.txt', "w") as f:
                for s in self.layer_importance:
                    f.write(str(s) + "\n")
            with open(save_dir / 'layer_pruned.txt', "w") as f:
                for s in self.selected_candidate:
                    f.write(str(s) + "\n")

        return self.selected_candidate

    def exec(self, verbose=True, model_path=None):
        model_to_prune = copy.deepcopy(self.model)
        for name, importance in self.selected_candidate:
            names = name.split('.')
            block, layer = int(names[1]), int(names[3])
            model_to_prune.model[block].m[layer] = nn.Identity()
            if verbose:
                print(name + ' pruned. Importance: ', importance)

        if model_path:
            print("LayerPruner start save model...")
            ckpt = {
                'epoch': -1,
                'best_fitness': None,
                'model': model_to_prune,
                'ema': None,
                'updates': None,
                'optimizer': None,
                'wandb_id': None, 
                'data': datetime.now().isoformat()
            }
            torch.save(ckpt, model_path)
            if verbose:
                print('Model saved to ', model_path)

        return model_to_prune


@ModifierYAML()
class LayerPrunerYOLOv3(LayerPruner):
    def set_model(self, model):
        self.model = model
        if self.choices:
            for name, m in model.named_modules():
                if isinstance(m, self.pruning_unit):
                    choice = False
                    for ch in self.choices:
                        if re.match(ch, name):
                            choice = True
                            break
                    if choice:
                        self.candidate.append([name, m])
        else:
            for name, m in model.named_modules():
                if isinstance(m, self.pruning_unit):
                    choice = True
                    for ignore in self.ignores:
                        if re.match(ignore, name):
                            choice = False
                            break
                    if choice:
                        self.candidate.append([name, m])

        if self.sparsity:
            weight_list = []
            for name, m in self.candidate:
                for module in m.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        weight_list.append(module.weight.data.abs().clone())

            all_bn_weights = torch.cat(weight_list)
            k = int(all_bn_weights.shape[0] * self.sparsity)
            global_threshold = torch.topk(
                all_bn_weights.view(-1), k, largest=False)[0].max()
            self.threshold = global_threshold

        self.calc_importance()

    def exec(self, verbose=True, model_path=None):
        model_to_prune = copy.deepcopy(self.model)
        for name, importance in self.selected_candidate:
            names = name.split('.')
            block, layer = int(names[1]), int(names[2])
            model_to_prune.model[block][layer] = nn.Identity()
            if verbose:
                print(name + ' pruned. Importance: ', importance)

        if model_path:
            print("Start save model...")
            torch.save(model_to_prune, model_path)
            if verbose:
                print('Model saved to ', model_path)

        return model_to_prune


def find_threshold(w_abs, bins=100, bound=1.5):
    threshold = 0.

    hist_y, hist_x = np.histogram(w_abs.cpu(), bins=bins, range=(0, bound))
    hist_y_diff = np.diff(hist_y)
    for i in range(len(hist_y_diff) - 1):
        if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
            threshold = hist_x[i + 1]
            return threshold

    return threshold


class FilterPruner(Pruner):
    def __init__(self, sparsity=0.5, **kwargs):
        super().__init__(**kwargs)

        self.sparsity = sparsity

        self.model = None
        self.candidate = []

        self.candidate_mask = {}

    def set_model(self, model):
        self.model = model
        if self.choices:
            for name, m in model.named_modules():
                if name in self.choices and isinstance(m, nn.Conv2d):
                    self.candidate.append([name, m])
        else:
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    choice = True
                    for ignore in self.ignores:
                        if re.match(ignore, name):
                            choice = False
                            break
                    if choice:
                        self.candidate.append([name, m])

    def compress(self):
        # self.candidate_mask = {}
        for name, m in self.candidate:
            filters = m.weight.data.shape[0]
            num_prune = int(filters * self.sparsity)
            mask = self.calc_mask(m, num_prune)
            self.candidate_mask[name] = mask

        return self.candidate_mask

    def exec(self, img_size=640, verbose=True, model_path=None):
        model_to_prune = copy.deepcopy(self.model)
        model_to_prune.model[-1].export = True

        example_inputs = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        dependency_graph = DependencyGraph().build_dependency(model_to_prune, example_inputs=example_inputs)

        for name, module in model_to_prune.named_modules():
            if name in self.candidate_mask.keys():
                prune_index = torch.where(self.candidate_mask[name]["weight_mask"].cpu() < 0.5)[0].numpy().tolist()
                if prune_index:
                    plan = dependency_graph.get_pruning_plan(module, prune_conv, prune_index)
                    plan.exec()

        model_to_prune.model[-1].export = False
        if model_path:
            torch.save(model_to_prune, model_path)
            if verbose:
                print('Pruned model saved to ', model_path)

        return model_to_prune


@ModifierYAML()
class L1FilterPruner(FilterPruner):

    @staticmethod
    def get_channel_sum(weight):
        filters = weight.shape[0]
        w_abs = weight.abs()
        w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
        return w_abs_structured

    def calc_mask(self, conv, num_prune):
        assert isinstance(conv, nn.Conv2d)

        weight = conv.weight.data.clone()
        # get the l1-norm sum for each filter
        w_abs_structured = self.get_channel_sum(weight)

        threshold = torch.topk(w_abs_structured.view(-1),
                               num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_abs_structured, threshold).type_as(weight)
        mask_bias = mask_weight.clone()
        mask = {'weight_mask': mask_weight.detach(),
                'bias_mask': mask_bias.detach()}

        return mask


@ModifierYAML()
class L2FilterPruner(FilterPruner):

    @staticmethod
    def get_channel_sum(weight):
        filters = weight.shape[0]
        w = weight.view(filters, -1)
        w_l2_norm = torch.sqrt((w ** 2).sum(dim=1))
        return w_l2_norm

    def calc_mask(self, conv, num_prune):
        assert isinstance(conv, nn.Conv2d)

        weight = conv.weight.data.clone()
        # get the l2-norm sum for each filter
        w_l2_norm = self.get_channel_sum(weight)

        threshold = torch.topk(w_l2_norm.view(-1),
                               num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_l2_norm, threshold).type_as(weight)
        mask_bias = mask_weight.clone()
        mask = {'weight_mask': mask_weight.detach(),
                'bias_mask': mask_bias.detach()}

        return mask


@ModifierYAML()
class FPGMFilterPruner(FilterPruner):

    @staticmethod
    def _get_distance_sum(weight, out_idx):
        """
        Calculate the total distance between a specified filter (by out_idex and in_idx) and
        all other filters.
        Parameters
        ----------
        weight: Tensor
            convolutional filter weight
        out_idx: int
            output channel index of specified filter, this method calculates the total distance
            between this specified filter and all other filters.
        Returns
        -------
        float32
            The total distance
        """
        # logger.debug('weight size: %s', weight.size())
        assert len(weight.size()) in [3, 4], 'unsupported weight shape'

        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()

    def get_channel_sum(self, weight):
        # weight = wrapper.module.weight.data

        assert len(weight.size()) in [3, 4]
        dist_list = []
        for out_i in range(weight.size(0)):
            dist_sum = self._get_distance_sum(weight, out_i)
            dist_list.append(dist_sum)
        return torch.Tensor(dist_list).to(weight.device)

    def _get_min_gm_kernel_idx(self, num_prune, weight):
        channel_dist = self.get_channel_sum(weight)

        dist_list = [(channel_dist[i], i)
                     for i in range(channel_dist.size(0))]
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:num_prune]

        return [x[1] for x in min_gm_kernels]

    def calc_mask(self, conv, num_prune):
        assert isinstance(conv, nn.Conv2d)

        weight = conv.weight.data.clone()
        filters = weight.data.shape[0]

        min_gm_idx = self._get_min_gm_kernel_idx(
            num_prune, weight)

        mask_weight = torch.ones(filters).type_as(weight)
        for idx in min_gm_idx:
            mask_weight[idx] = 0.
        mask_bias = mask_weight.clone()
        mask = {'weight_mask': mask_weight.detach(),
                'bias_mask': mask_bias.detach()}

        return mask


@ModifierYAML()
class TaylorFOWeightFilterPruner(FilterPruner):

    @staticmethod
    def calc_contributions(weight):
        """
        Calculate the estimated importance of filters as a sum of individual contribution
        based on the first order taylor expansion.
        """

        filters = weight.size(0)
        contribution = (
                weight * weight.grad).data.pow(2).view(filters, -1).sum(dim=1)
        return contribution

    def calc_mask(self, conv, num_prune):
        assert isinstance(conv, nn.Conv2d)

        weight = conv.weight.data.clone()
        filters = weight.data.shape[0]

        channel_contribution = self.calc_contributions(weight)
        prune_indices = torch.argsort(channel_contribution)[:num_prune]

        mask_weight = torch.ones(filters).type_as(weight)
        for idx in prune_indices:
            mask_weight[idx] = 0.
        mask_bias = mask_weight.clone()
        mask = {'weight_mask': mask_weight.detach(),
                'bias_mask': mask_bias.detach()}

        return mask


@ModifierYAML()
class SlimPruner(Pruner):
    def __init__(self, sparsity, **kwargs):
        super().__init__(**kwargs)

        self.sparsity = sparsity

        self.model = None
        self.candidate = []

        self.candidate_mask = {}

    def set_model(self, model):
        self.model = model
        if self.choices:
            for name, m in model.named_modules():
                if name in self.choices and isinstance(m, nn.BatchNorm2d):
                    self.candidate.append([name, m])
        else:
            for name, m in model.named_modules():
                # if name not in self.ignores and isinstance(m, nn.BatchNorm2d):
                #     self.candidate.append([name, m])
                if isinstance(m, nn.BatchNorm2d):
                    choice = True
                    for ignore in self.ignores:
                        if re.match(ignore, name):
                            choice = False
                            break
                    if choice:
                        self.candidate.append([name, m])

    @staticmethod
    def calc_mask(bn, threshold, layer_keep=0.1):
        assert isinstance(bn, nn.BatchNorm2d)

        weight = bn.weight.data.clone()

        base_mask = torch.ones(weight.size()).type_as(weight).detach()
        mask = {'weight_mask': base_mask.detach(),
                'bias_mask': base_mask.clone().detach()}
        filters = weight.size(0)

        # num_prune = int(filters * sparsity)

        # if filters >= 2 and num_prune >= 1:
        if filters >= 2:
            w_abs = weight.abs()
            # mask_weight = torch.gt(w_abs, self.global_threshold).type_as(weight)
            mask_weight = torch.gt(w_abs, threshold).type_as(weight)
            if mask_weight.sum().item() < filters * layer_keep:
                k = int(weight.shape[0] * (1 - layer_keep))
                keep_threshold = torch.topk(
                    weight.view(-1), k, largest=False)[0].max()
                mask_weight = torch.gt(w_abs, keep_threshold).type_as(weight)
            mask_bias = mask_weight.clone()
            mask = {'weight_mask': mask_weight.detach(),
                    'bias_mask': mask_bias.detach()}

        return mask

    def compress(self):
        weight_list = []
        for name, layer in self.candidate:
            weight_list.append(layer.weight.data.abs().clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * self.sparsity)
        global_threshold = torch.topk(
            all_bn_weights.view(-1), k, largest=False)[0].max()

        for name, m in self.candidate:
            # filters = m.weight.data.shape[0]
            # num_prune = int(filters * self.sparsity)
            mask = self.calc_mask(m, global_threshold)
            self.candidate_mask[name] = mask

        return self.candidate_mask

    def exec(self, img_size=640, verbose=True, model_path=None):
        model_to_prune = copy.deepcopy(self.model)
        model_to_prune.model[-1].export = True

        example_inputs = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        dependency_graph = DependencyGraph().build_dependency(model_to_prune, example_inputs=example_inputs)

        for name, module in model_to_prune.named_modules():
            if name in self.candidate_mask.keys():
                prune_index = torch.where(self.candidate_mask[name]["weight_mask"].cpu() < 0.5)[0].numpy().tolist()
                if prune_index:
                    plan = dependency_graph.get_pruning_plan(module, prune_batchnorm, prune_index)
                    plan.exec()

        model_to_prune.model[-1].export = False
        if model_path:
            print("SlimPruner start save model...")
            torch.save(model_to_prune, model_path)
            if verbose:
                print('Model saved to ', model_path)

        return model_to_prune


@ModifierYAML()
class ChannelPruner(Pruner):
    def __init__(self, ignores=None, bins=100, upper_bound=1.0, **kwargs):
        super().__init__(**kwargs)

        self.ignores = ignores
        self.bins = bins
        self.upper_bound = upper_bound

        self.model = None
        self.candidate = []

        self.candidate_mask = {}
        self.global_threshold = 0.
        self.adaptive_threshold = 0.
        self.layer_threshold = {}

    def set_model(self, model):
        self.model = model
        # for name, m in model.named_modules():
        #     if name not in self.ignores and isinstance(m, nn.BatchNorm2d):
        #         # weight_list.append(m.weight.data.abs().clone())
        #         self.candidate.append([name, m])
        for name, m in model.named_modules():
            # if isinstance(m, nn.BatchNorm2d) and name not in ignores:
            #     bn_modules.append(m)
            if isinstance(m, nn.BatchNorm2d):
                choice = True
                for ignore in self.ignores:
                    if re.match(ignore, name):
                        choice = False
                        break
                if choice:
                    self.candidate.append([name, m])

    def calc_threshold(self, sparsity=None):
        if sparsity is None:
            sparsity = 0.4

        weight_list = []
        for name, layer in self.candidate:
            weight_list.append(layer.weight.data.abs().clone())
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * sparsity)
        self.global_threshold = torch.topk(
            all_bn_weights.view(-1), k, largest=False)[0].max()

        self.adaptive_threshold = find_threshold(all_bn_weights, bins=self.bins, bound=self.upper_bound)

        for name, m in self.candidate:
            threshold = find_threshold(m.weight.data.abs().clone())
            self.layer_threshold[name] = threshold

    @staticmethod
    def calc_mask(bn, threshold, layer_keep=0.1):
        assert isinstance(bn, nn.BatchNorm2d)

        weight = bn.weight.data.clone()

        base_mask = torch.ones(weight.size()).type_as(weight).detach()
        mask = {'weight_mask': base_mask.detach(),
                'bias_mask': base_mask.clone().detach()}
        filters = weight.size(0)

        # num_prune = int(filters * sparsity)

        # if filters >= 2 and num_prune >= 1:
        if filters >= 2:
            w_abs = weight.abs()
            # mask_weight = torch.gt(w_abs, self.global_threshold).type_as(weight)
            mask_weight = torch.gt(w_abs, threshold).type_as(weight)
            if mask_weight.sum().item() < 0.5:
                k = int(weight.shape[0] * (1 - layer_keep))
                keep_threshold = torch.topk(
                    weight.view(-1), k, largest=False)[0].max()
                mask_weight = torch.gt(w_abs, keep_threshold).type_as(weight)
            mask_bias = mask_weight.clone()
            mask = {'weight_mask': mask_weight.detach(),
                    'bias_mask': mask_bias.detach()}

        return mask

    def compress(self, threshold=None, sparsity=None, layer_wise=True, save_dir=None):
        """
        """

        self.calc_threshold(sparsity=sparsity)

        self.candidate_mask = {}
        for name, m in self.candidate:
            if threshold:
                mask = self.calc_mask(m, threshold=threshold)
            elif sparsity:
                mask = self.calc_mask(m, threshold=self.global_threshold)
            elif layer_wise:
                mask = self.calc_mask(m, threshold=self.layer_threshold[name])
            else:
                mask = self.calc_mask(m, threshold=self.adaptive_threshold)

            self.candidate_mask[name] = mask

        if save_dir:
            with open(save_dir / 'channel_to_prune.txt', "w") as f:
                for (name, mask) in self.candidate_mask.items():
                    mask = mask['weight_mask']
                    width = mask.shape[0]
                    pruned = width - int(mask.sum().item())
                    if pruned > 0:
                        f.write(name + ":" + str(pruned) + "/" + str(width) + "\n")

        return self.candidate_mask

    def exec(self, img_size=640, verbose=True, model_path=None):
        """
        speedup
        :return:
        """
        model_to_prune = copy.deepcopy(self.model)
        model_to_prune.model[-1].export = True

        example_inputs = torch.randn(1, 3, img_size, img_size, requires_grad=True)
        dependency_graph = DependencyGraph().build_dependency(model_to_prune, example_inputs=example_inputs)

        for name, module in model_to_prune.named_modules():
            # skipped = ('model.2' in name) or ('model.4' in name) or ('model.6' in name)
            # if name in self.candidate_mask.keys() and not ('.m.' in name and 'cv2' in name and skipped):
            if name in self.candidate_mask.keys():
                prune_index = torch.where(self.candidate_mask[name]["weight_mask"].cpu() < 0.5)[0].numpy().tolist()
                if prune_index:
                    plan = dependency_graph.get_pruning_plan(module, prune_batchnorm, prune_index)
                    plan.exec()
                    # if verbose:
                    #     print(plan)

        model_to_prune.model[-1].export = False
        if model_path:
            print("ChanelPruner start save model...")
            ckpt = {
                'epoch': -1,
                'best_fitness': None,
                'model': model_to_prune,
                'ema': None,
                'updates': None,
                'optimizer': None,
                'wandb_id': None, 
                'data': datetime.now().isoformat()
            }
            torch.save(ckpt, model_path)
            if verbose:
                print('Model saved to ', model_path)

        return model_to_prune
