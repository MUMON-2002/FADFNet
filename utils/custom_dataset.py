import numpy as np
from torch.utils.data import Dataset


class Custom_Dataset(Dataset):

    def __init__(self,
                 input_list=None, 
                 target_list=None, 
                 patch_n=None, 
                 patch_size=None
                 ):

        self.input_ = input_list
        self.target_ = target_list
        self.patch_n = patch_n
        self.patch_size = patch_size

    def __len__(self):
        return len(self.target_)

    def get_patch(self, full_input_img, full_target_img):
        assert full_input_img.shape == full_target_img.shape
        h, w = full_input_img.shape
        p = self.patch_size

        pad_h = max(0, p - h)
        pad_w = max(0, p - w)
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            padded_input = np.pad(full_input_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            padded_target = np.pad(full_target_img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

            current_h, current_w = padded_input.shape
        else:
            padded_input = full_input_img
            padded_target = full_target_img

            current_h, current_w = h, w

        if current_h == p and current_w == p:
            patch_input_imgs = [padded_input] * self.patch_n
            patch_target_imgs = [padded_target] * self.patch_n

            return np.stack(patch_input_imgs), np.stack(patch_target_imgs)

        patch_input_imgs, patch_target_imgs = [], []
        sampled_coords = set()
        max_attempts = self.patch_n * 10
        attempts = 0

        while len(patch_input_imgs) < self.patch_n and attempts < max_attempts:
            top = np.random.randint(0, current_h - p + 1)
            left = np.random.randint(0, current_w - p + 1)
            coord = (top, left)

            if coord not in sampled_coords:
                sampled_coords.add(coord)
                patch_input_imgs.append(padded_input[top:top + p, left:left + p])
                patch_target_imgs.append(padded_target[top:top + p, left:left + p])

            attempts += 1

        assert len(patch_input_imgs) == self.patch_n

        return np.stack(patch_input_imgs), np.stack(patch_target_imgs)

    def __getitem__(self, idx):

        input_img = np.load(self.input_[idx]).astype(np.float32)
        target_img = np.load(self.target_[idx]).astype(np.float32)

        input_patches, target_patches = self.get_patch(input_img, target_img)

        input_patches = np.expand_dims(input_patches, axis=1)
        target_patches = np.expand_dims(target_patches, axis=1) # (patch_n, dim=1, patch_size, patch_size)

        assert input_patches.shape[0] == self.patch_n
        return input_patches, target_patches