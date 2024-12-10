from pathlib import Path

import numpy as np
import torch.utils.data
from tqdm import tqdm

from metric.milvue_util import getClient, topk
from metric.model import create_model
from my import MyDataset

collection = 'UKBench_mynet'
root = r'D:\dataset\ukbench\full'

if __name__ == '__main__':
    model, preprocessor = create_model('../model.pth')

    client = getClient()
    src_paths = list(Path(root).glob('**/*.jpg'))[:8]

    dataset = MyDataset(src_paths, preprocessor)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, drop_last=False)

    record = np.zeros((len(src_paths), 4))
    cnt = 0
    for img_idx, img in tqdm(dataLoader, total=len(dataLoader)):
        img = img.cuda()
        with torch.no_grad():
            f = model(img).cpu().numpy().tolist()

        res = topk(client, collection, f, 4)
        for i in range(len(res)):
            query = img_idx[i].item()
            print(res[i])
            for j in range(len(res[i])):
                hit = res[i][j]
                match = int(hit['id'])
                if query // 4 == match // 4:
                    record[query, j] = 1
                    cnt += 1
    np.save('query_resnet50.npy', record)
    print(float(cnt / len(src_paths)))