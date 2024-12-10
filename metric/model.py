import torch
import torchvision.models as models
from my import MyNet
from torchvision.transforms import transforms

def create_model(weight: str):

    model = MyNet(models.__dict__['resnet50']).cuda().eval()
    my_dict = torch.load(weight)
    model.load_state_dict(my_dict)

    preprocessor = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            ),
        ]
    )

    return model, preprocessor