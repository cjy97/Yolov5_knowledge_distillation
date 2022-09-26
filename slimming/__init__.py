from .utils import *
from .regularizer import *
from .pruner import *


def load_regularizer(yaml_file):
    """
    :param yaml_file: a string representation of the yaml syntax to load a regularizer
    :return: the loaded regularizer
    """
    yaml_str = load_recipe_yaml_str(yaml_file)
    container = yaml.safe_load(yaml_str)

    return container['regularizers'][0]


def load_pruners(yaml_file):
    """
    :param yaml_file: a string representation of the yaml syntax to load a regularizer
    :return: the loaded regularizer
    """
    yaml_str = load_recipe_yaml_str(yaml_file)
    container = yaml.safe_load(yaml_str)

    return container['pruners']
