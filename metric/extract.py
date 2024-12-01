from pathlib import Path

import torch.utils.data
from tqdm import tqdm

from metric.milvue_util import getClient, insert
from metric.model import create_model
from my import MyDataset

collection = 'UKBench_mynet'
root = r'D:\dataset\ukbench\full'

if __name__ == '__main__':

    model, preprocessor = create_model('../model.pth')

    client = getClient()
    src_paths = list(Path(root).glob('**/*.jpg'))

    dataset = MyDataset(src_paths, preprocessor)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, drop_last=False)
    for img_name, img in tqdm(dataLoader, total=len(dataLoader)):
        img = img.cuda()
        with torch.no_grad():
            f = model(img).cpu().numpy().tolist()
        data = []
        for i in range(len(img_name)):
            data.append({'pk': img_name[i].item(), 'embeddings': f[i]})
        insert(client, collection, data)
