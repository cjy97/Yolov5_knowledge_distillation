import yaml


def load_recipe_yaml_str(file_path) -> str:
    """
    Loads a YAML recipe file to a string

    YAML front matter: https://jekyllrb.com/docs/front-matter/

    :param file_path: file path to recipe YAML file
    :return: the recipe YAML configuration loaded as a string
    """

    extension = file_path.lower().split(".")[-1]
    if extension not in ["yaml"]:
        raise ValueError(
            "Unsupported file extension for recipe. Excepted '.yaml'. "
            "Received {}".format(file_path)
        )
    with open(file_path, "r") as yaml_file:
        yaml_str = yaml_file.read()

    return yaml_str


class ModifierYAML(object):
    """
    A decorator to handle making a modifier class YAML ready.
    IE it can be loaded in through the yaml plugin easily.

    """

    def __call__(self, clazz):
        """
        :param clazz: the class to create yaml constructors for
        :return: the class after yaml constructors have been added
        """
        yaml_key = "!{}".format(clazz.__name__)

        def constructor(loader, node):
            instance = clazz.__new__(clazz)
            yield instance
            state = loader.construct_mapping(node, deep=True)
            instance.__init__(**state)

        yaml.add_constructor(yaml_key, constructor)
        yaml.add_constructor(yaml_key,
                             constructor,
                             yaml.SafeLoader, )

        return clazz
