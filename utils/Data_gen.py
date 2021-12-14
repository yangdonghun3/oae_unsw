from torch.utils.data import Dataset
import torch

class Data_gen(Dataset):
    def __init__(self, df):
        tmp = df.values
        self.data_torch = torch.from_numpy(tmp).type(torch.FloatTensor)
        
    def __len__(self):
        return self.data_torch.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_torch[idx,:]