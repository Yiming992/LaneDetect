from torch.utils.data import Dataset,DataLoader

class Tusimple_data(Dataset):

    def __init__(self,file,root_dir,transform=None):
        super(Tusimple_data).__init__()
        self.file=file
        self.root_dir=root_dir
        self.transform=transform 
    

    def __getitem__(self,index):
        pass
    
    def __len__(self):
        pass
 

class New_Data(Dataset):
    pass









