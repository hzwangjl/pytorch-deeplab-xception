from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image, ImageFilter

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import random
import sys
sys.path.append('../../')
from mypath import Path

class SegmentDataset(Dataset):
    NUM_CLASSES = 3
    def __init__(self, args, root=Path.db_root_dir('mydataset'), split="train", transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.args = args
        self.base_size = args.base_size
        self.crop_size = args.crop_size
        self.transform = transform
        if split == 'train':
            images_dir = root + 'train/images'
            masks_dir = root + 'train/masks'
        elif split == "valid":
            images_dir = root + 'valid/images'
            masks_dir = root + 'valid/masks'
        else:
            print(split, " not support")   
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.npy') for image_id in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = Image.open(self.images_fps[index])
        # raw_img = image
        mask = np.load(self.masks_fps[index])
        mask[mask > self.NUM_CLASSES - 1] = 0
        mask = Image.fromarray(mask)
        if self.split == "train":
            image, mask = self._sync_transform(image, mask)
        elif self.split == "valid":
            image, mask = self._val_sync_transform(image, mask)
        mask = np.array(mask)

        if self.transform is not None:
            image = self.transform(image)
            #raw_img = self.transform(raw_img)

        # return raw_img, image, mask
        return image, mask

    def _sync_transform(self, img, mask):
        w, h = img.size
        crop_size = self.crop_size

        # Center Crop
        if w != h:
            img = img.crop((w/2 - h/2, 0, w/2 + h/2, h))
            mask = mask.crop((w/2 - h/2, 0, w/2 + h/2, h))

        # resize & ramdom crop
        w, h = img.size
        if w != crop_size:
            if random.random() < 0.2:
                x1 = random.randint(0, w - crop_size)
                x2 = x1 + crop_size
                y1 = random.randint(0, h - crop_size)
                y2 = y1 + crop_size
                img = img.crop((x1, y1, x2, y2))
                mask = mask.crop((x1, y1, x2, y2))
            else:
                ow = crop_size
                oh = crop_size
                img = img.resize((ow, oh), Image.BILINEAR)
                mask = mask.resize((ow, oh), Image.NEAREST)

        # Scale up
        if random.random() < 0.2:
            w, h = img.size
            ratio = random.uniform(0.7, 0.9)
            ow = int(w*ratio)
            oh = int(h*ratio)
            img = img.crop((w/2 - ow/2, h/2-oh/2, w/2 + ow/2, h/2+oh/2))
            mask = mask.crop((w/2 - ow/2, h/2-oh/2, w/2 + ow/2, h/2+oh/2))
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)

        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random mirror
        if random.random() < 0.1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # random rotate
        if random.random() < 0.3:
            angle = random.randint(-60, 60)
            img = img.rotate(angle)
            mask = mask.rotate(angle)

        # random rotate
        # if random.random() < 0.2:
        #     img = img.transpose(Image.ROTATE_90)
        #     mask = mask.transpose(Image.ROTATE_90)

        # if random.random() < 0.2:
        #     img = img.transpose(Image.ROTATE_270)
        #     mask = mask.transpose(Image.ROTATE_270)

        # gaussian blur as in PSP
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img, mask

    def _val_sync_transform(self, img, mask):

        w, h = img.size
        crop_size = self.crop_size

        # Center Crop
        if w != h:
            img = img.crop((w/2 - h/2, 0, w/2 + h/2, h))
            mask = mask.crop((w/2 - h/2, 0, w/2 + h/2, h))

        # resize & ramdom crop
        w, h = img.size
        if w != crop_size:
            ow = crop_size
            oh = crop_size
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)

        return img, mask

def mask_to_image(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = Image.new('RGB', (w, h))
    for j in range(0, h):
        for i in range(0, w):
            pixal = mask[j, i]
            if pixal == 0:
                mask_rgb.putpixel((i, j), (61,10,81))
            elif pixal == 1:
                mask_rgb.putpixel((i, j), (69,142,139))
            elif pixal == 2:
                mask_rgb.putpixel((i, j), (250,231,85))
            else:
                mask_rgb.putpixel((i, j), (255, 255, 255))
    return mask_rgb

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 240
    args.crop_size = 240

    train_transform=transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        # transforms.Normalize([0.519401, 0.359217, 0.310136], [0.061113, 0.048637, 0.041166])
    ])
    valid_transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.517446, 0.360147, 0.310427], [0.061526,0.049087, 0.041330])
    ])

    train_set = SegmentDataset(args, split='train', transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)

    valid_set = SegmentDataset(args, split='valid', transform=valid_transform)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=8)

    for batch_idx, data in enumerate(train_loader):
        img = data[0][0].numpy()*255
        img = img.astype('uint8')
        img = np.transpose(img,(1,2,0))

        mask = data[1][0]
        mask[mask > train_set.NUM_CLASSES - 1] = 0
        mask = mask_to_image(mask)

        plt.subplot(1, 2, 1)
        plt.title('image')
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.imshow(mask)

        save_file = "train/"
        os.makedirs(save_file, exist_ok=True)
        plt.savefig(os.path.join(save_file, str(batch_idx) + '.png'))
        # plt.show()

        if batch_idx % 10 == 0:
            print("### proc train %d / %d"%(batch_idx, len(train_loader)))


    for batch_idx, data in enumerate(valid_loader):
        img = data[0][0].numpy()*255
        img = img.astype('uint8')
        img = np.transpose(img,(1,2,0))
        img = Image.fromarray(img)

        mask = data[1][0]
        mask[mask > valid_set.NUM_CLASSES - 1] = 0
        mask = mask_to_image(mask)

        plt.subplot(1, 2, 1)
        plt.title('image')
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title('mask')
        plt.imshow(mask)

        save_file = "val/"
        os.makedirs(save_file, exist_ok=True)
        plt.savefig(os.path.join(save_file, str(batch_idx) + '.png'))

        if batch_idx % 100 == 0:
            print("### proc val %d / %d"%(batch_idx, len(valid_loader)))
