# 制定自己的dataset
import pandas as pd
from torch.utils.data import Dataset

class StoryDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file)
        self.data_df = self.data_df[["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]]
        self.data_df = self.data_df.values
        self.data_df = self.data_df.astype(str)
        self.data_df = self.data_df.tolist()
        # 把每个故事的句子合并
        for i in range(len(self.data_df)):
            self.data_df[i] = ' '.join(self.data_df[i])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df[idx]
    
