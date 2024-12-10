import torch
from PIL import Image

from metric.model import create_model

if __name__ == '__main__':
    model, preprocessor = create_model('../model.pth')

    url = r'D:\dataset\ukbench\full\ukbench00000.jpg'
    img = Image.open(url)
    x = preprocessor(img).unsqueeze(0).cuda()
    y = model(x)

    url1 = r'D:\dataset\ukbench\full\ukbench01015.jpg'
    img1 = Image.open(url1)
    x1 = preprocessor(img1).unsqueeze(0).cuda()
    y1 = model(x1)

    print(torch.nn.functional.cosine_similarity(y, y1, dim=1))