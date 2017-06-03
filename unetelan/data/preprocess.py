import glob
import re

import numpy.random as npr
import torch as th
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import *

np.random.seed(0)


class HRFLoader:
    def __init__(self, path, h=True, g=True, dr=True):
        code = ['h'] * h + ['g'] * g + ['dr'] * dr
        code = "|".join(code)
        code = f'(.*)(\d\d_)({code}).jpg'

        candidates = glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.JPG')
        self.filenames = [fname for fname in candidates if re.match(code, fname, flags=re.IGNORECASE)]

    def __call__(self):
        images = []
        masks = []

        for fname in self.filenames:
            images.append(Image.open(fname))
            masks.append(Image.open(fname.replace("JPG", "jpg").replace("jpg", "tif")))

        return images, masks


class Rotate:
    def __init__(self, n, max_angle):
        self.n = n
        self.max_angle = max_angle

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []

        for img, msk in zip(images, masks):
            ret_images += [img]
            ret_masks += [msk]

            for _ in range(self.n):
                alpha = npr.rand() * self.max_angle

                ret_images += [img.rotate(alpha, Image.BICUBIC)]
                ret_masks += [msk.rotate(alpha, Image.BICUBIC)]

        return ret_images, ret_masks


class RandomCrop:
    def __init__(self, n, size, th_img=50, th_msk=50):
        self.size = size
        self.n = n
        self.th_img = th_img * 255
        self.th_msk = th_msk * 255

    def crop(self, img, msk):
        w, h = img.size
        nnz_msk = 0
        nnz_img = 0

        max_left = w - self.size
        max_upper = h - self.size

        img_crop, msk_crop = None, None
        while nnz_img < self.th_img or nnz_msk < self.th_msk:
            left = npr.randint(low=0, high=max_left, size=1)
            upper = npr.randint(low=0, high=max_upper, size=1)

            box = (left, upper, left + self.size, upper + self.size)  # left, upper, right, and lower
            box = tuple(map(int, box))

            img_crop, msk_crop = img.crop(box).copy(), msk.crop(box).copy()

            nnz_img = np.sum(img_crop)
            nnz_msk = np.sum(msk_crop)

        return img_crop, msk_crop

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []

        for img, msk in zip(images, masks):
            for _ in range(self.n):
                img_crop, msk_crop = self.crop(img, msk)

                ret_images.append(img_crop)
                ret_masks.append(msk_crop)

        return ret_images, ret_masks


def RGBConverter():
    def f(images, masks):
        for i, img in enumerate(images):
            images[i] = img.convert(mode='L')

        return images, masks

    return f


def ToNumpy():
    def f(images, masks):
        for i, (img, msk) in enumerate(zip(images, masks)):
            img = np.array(img, dtype=np.float32)
            img /= 255.

            msk = np.array(msk, dtype=np.float32)
            msk /= 255

            images[i] = img
            masks[i] = msk

        return images, masks

    return f


def FeatureWiseStd(with_mean=True, with_std=True):
    def f(images, masks):
        ret_images = []
        for img in images:
            img -= np.mean(img, axis=2, keepdims=True)
            img /= np.std(img, axis=2, keepdims=True) + 1e-6

            ret_images.append((img))

        return ret_images, masks

    return f


def ChannelWiseStd(with_mean=True, with_std=True):
    def f(images, masks):
        ret_images = []
        for img in images:
            img -= np.mean(img, axis=(0, 1), keepdims=True)
            img /= np.std(img, axis=(0, 1), keepdims=True) + 1e-6

            ret_images.append((img))

        return ret_images, masks

    return f


def Standardizer(with_mean=True, with_std=True):
    def f(images, masks):
        mean = np.zeros(images[0].shape, dtype=np.float64)
        std = np.zeros(images[0].shape, dtype=np.float64)

        for img in images:
            mean += img
        mean /= len(images)

        for img in images:
            std += (img - mean) ** 2
        std /= len(images)
        std = np.sqrt(std)

        if with_mean:
            for i, img in enumerate(images):
                img = (img)
                img -= mean
                images[i] = (img)

        if with_std:
            for i, img in enumerate(images):
                img = (img)
                img /= std + 1e-6
                images[i] = (img)

        return images, masks

    return f


class ListDataset(th.utils.data.Dataset):
    def __init__(self, images, masks):
        assert len(images) == len(masks)

        self.images = images
        self.masks = masks

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    batch = [(img, msk.type(th.FloatTensor)) for img, msk in batch]
    images, masks = default_collate(batch)

    return images, masks


