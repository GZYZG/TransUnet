import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk


def nii2slices(nii_path):
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)

    return array


def gen_train_slices(axis='Z'):
    import re
    train_dir = "../../abdominal-multi-organ-segmentation/dataset/prep_train"
    ct = os.path.join(train_dir, "CT")
    gt = os.path.join(train_dir, "GT")

    files = os.listdir(ct)
    train = []
    case_no = re.compile(r"\d{4}")

    if axis != 'Z':
        save_dir = f"../data/Synapse_{axis}/train_npz"
    else:
        save_dir = f"../data/Synapse/train_npz"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for f in files:
        ct_slices = nii2slices(os.path.join(ct, f))
        gt_slices = nii2slices(os.path.join(gt, f.replace("img", "label")))
        assert len(ct_slices) == len(gt_slices)

        if axis == 'X':
            ct_slices = np.transpose(ct_slices, (1, 0, 2))
            gt_slices = np.transpose(gt_slices, (1, 0, 2))
        elif axis == 'Y':
            ct_slices = np.transpose(ct_slices, (2, 0, 1))
            gt_slices = np.transpose(gt_slices, (2, 0, 1))

        for i, (cslice, gslice) in enumerate(zip(ct_slices, gt_slices)):
            no = case_no.findall(f)[0]
            np.savez(os.path.join(save_dir, f"case{no}_slice{i:0>4}.npz"), image=cslice, label=gslice)
            train.append(f"case{no}_slice{i:0>4}\n")
        print(f"{f} with {i+1} slices are processed ...")

    if axis != 'Z':
        list_dir = f"../lists/lists_Synapse_{axis}"
    else:
        list_dir = f"../lists/lists_Synapse"

    with open(os.path.join(list_dir, "train.txt"), 'w') as f:
        f.writelines(train)
    # print(train.__len__())
    print(f"slice generated in {list_dir}/train.txt")


def gen_test_data(axis='Z'):
    import re
    import h5py
    test_dir = "../../abdominal-multi-organ-segmentation/dataset/prep_val"
    ct_dir = os.path.join(test_dir, "CT")
    gt_dir = os.path.join(test_dir, "GT")

    files = os.listdir(ct_dir)
    test = []
    case_no = re.compile(r"\d{4}")
    if axis != 'Z':
        save_dir = f"../data/Synapse_{axis}/test_vol_h5"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, f in enumerate(files):
        ct_slices = nii2slices(os.path.join(ct_dir, f))
        gt_slices = nii2slices(os.path.join(gt_dir, f.replace("img", "label")))
        assert len(ct_slices) == len(gt_slices)

        image = sitk.ReadImage(os.path.join(ct_dir, f))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        if axis == 'X':  # (z, x, y) => (x, z, y)
            dest = [1, 0, 2]
            ct_slices = np.transpose(ct_slices, dest)
            gt_slices = np.transpose(gt_slices, dest)
            spacing = tuple(np.array(spacing)[dest])
            origin = tuple(np.array(origin)[dest])
            direction = tuple(np.array(direction)[dest])
        elif axis == 'Y':  # (z, x, y) => (y, z, x)
            dest = [2, 0, 1]
            ct_slices = np.transpose(ct_slices, dest)
            gt_slices = np.transpose(gt_slices, dest)
            spacing = tuple(np.array(spacing)[dest])
            origin = tuple(np.array(origin)[dest])
            direction = tuple(np.array(direction)[dest])

        no = case_no.findall(f)[0]
        case = h5py.File(os.path.join(save_dir, f"case{no:0>4}.npy.h5"), 'w')
        case.create_dataset('image', data=ct_slices, shape=ct_slices.shape, dtype=ct_slices.dtype)
        case.create_dataset('label', data=gt_slices, shape=gt_slices.shape, dtype=gt_slices.dtype)

        case.create_dataset('spacing', data=spacing)
        case.create_dataset('direction', data=direction)
        case.create_dataset('origin', data=origin)
        case.flush()
        case.close()
        print(f"{f} case are processed ...")
        test.append(f"case{no:0>4}\n")

    if axis != 'Z':
        list_dir = f"../lists/lists_Synapse_{axis}"
    else:
        list_dir = f"../lists/lists_Synapse"

    if not os.path.exists(list_dir):
        os.mkdir(list_dir)

    with open(os.path.join(list_dir, 'test_vol.txt'), "w") as f:
        f.writelines(test)

    print(f"Cases generated in {list_dir}/test_vol.txt")


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            # print(slice_name)
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath, 'r')
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.split != "train":
            sample['spacing'] = data['spacing'][:]
            sample['direction'] = data['direction'][:]
            sample['origin'] = data['origin'][:]
        return sample


class Masked_Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, visible_class=None, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.visible_class = visible_class  # 表示哪些标签是可见的

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            # print(slice_name)
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath, 'r')
            image, label = data['image'][:], data['label'][:]

        if self.visible_class is not None:
            _label = np.zeros_like(label)
            for cls in self.visible_class:
                _label[label == cls] = cls
            label = _label

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.split != "train":
            sample['spacing'] = data['spacing'][:]
            sample['direction'] = data['direction'][:]
            sample['origin'] = data['origin'][:]
        return sample


if __name__ == "__main__":
    # 生成训练数据集
    gen_train_slices(axis='X')
    # 生成测试数据集
    # gen_test_data(axis='X')
