import torch
from models.model_configs import (
    get_yolov7_config,
    get_yolov7_d6_config,
    get_yolov7_e6_config,
    get_yolov7_e6e_config,
    get_yolov7_tiny_config,
    get_yolov7_w6_config,
    get_yolov7x_config,
)
from models.yolo import Yolov7Model

MODEL_CONFIGS = {
    "yolov7": get_yolov7_config,
    "yolov7x": get_yolov7x_config,
    "yolov7-tiny": get_yolov7_tiny_config,
    "yolov7-w6": get_yolov7_w6_config,
    "yolov7-d6": get_yolov7_d6_config,
    "yolov7-e6": get_yolov7_e6_config,
    "yolov7-e6e": get_yolov7_e6e_config,
}


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def create_yolov7_model(
    architecture,
    num_classes=80,
    anchor_sizes_per_layer=None,
    num_channels=3,
    pretrained=True,
):
    config = MODEL_CONFIGS[architecture](
        num_classes=num_classes,
        anchor_sizes_per_layer=anchor_sizes_per_layer,
        num_channels=num_channels,
    )

    model = Yolov7Model(model_config=config)

    if pretrained:
        state_dict_path = config["state_dict_path"]
        if state_dict_path is None:
            raise ValueError(
                "Pretrained weights are not available for this architecture"
            )
        try:
            # load state dict
            state_dict = intersect_dicts(
                torch.hub.load_state_dict_from_url(state_dict_path, progress=False),
                model.state_dict(),
                exclude=["anchor"],
            )
            model.load_state_dict(state_dict, strict=False)
            print(
                f"Transferred {len(state_dict)}/{len(model.state_dict())} items from {state_dict_path}"
            )
        except Exception as e:
            print(f"Unable to load pretrained model weights from {state_dict_path}")
            print(e)
    return model
