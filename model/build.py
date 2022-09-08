from timm.models import create_model
from torch import nn


def build_model(config):
    model_name = config.model.name

    if config.model.pretrained:
        checkpoint_path = config.model.initial_checkpoint
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        model = create_model(model_name,
                             num_classes=1000,
                             drop_path_rate=config.aug.drop_path,
                             checkpoint_path=checkpoint_path)
        print("=> loaded pre-trained model '{}'".format(checkpoint_path))
        if model.num_classes:
            model.fc = nn.Linear(model.fc.in_features, config.model.num_classes)
    else:
        model = create_model(model_name,
                             num_classes=config.model.num_classes,
                             drop_path_rate=config.aug.drop_path)
    return model