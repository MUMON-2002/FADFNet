import numpy as np
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    def __init__(self, input_list=None, target_list=None, normalize=True, mode='train', patch_size=None, patch_num=1, context=False):    
        self.input_ = input_list
        self.target_ = target_list
        self.normalize = normalize
        self.mode = mode
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.context = context

    def __len__(self):
        return len(self.target_)
    
    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return img

    def __getitem__(self, idx):
        if self.context:
            path_list = self.input_[idx] 
            imgs = []
            for path in path_list:
                img = np.load(path).astype(np.float32)
                if self.normalize:
                    img = self.normalize_(img)
                imgs.append(img)
            input_img = np.stack(imgs, axis=0) 
        else:
            input_img = np.load(self.input_[idx]).astype(np.float32)
            if self.normalize:
                input_img = self.normalize_(input_img)
            input_img = np.expand_dims(input_img, axis=0) 

        target_path = self.target_[idx]
        target_img = np.load(target_path).astype(np.float32)
        if self.normalize:
            target_img = self.normalize_(target_img)
        target_img = np.expand_dims(target_img, axis=0) 

        if self.mode == 'train' and self.patch_size is not None:
            _, H, W = input_img.shape
            input_patches = []
            target_patches = []
            
            for _ in range(self.patch_num):
                top = np.random.randint(0, H - self.patch_size + 1)
                left = np.random.randint(0, W - self.patch_size + 1)
                
                input_patches.append(input_img[:, top:top + self.patch_size, left:left + self.patch_size])
                target_patches.append(target_img[:, top:top + self.patch_size, left:left + self.patch_size])
            
            input_img = np.stack(input_patches, axis=0)
            target_img = np.stack(target_patches, axis=0)

        return input_img, target_img
