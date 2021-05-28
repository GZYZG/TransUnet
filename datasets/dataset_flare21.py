import os
import glob
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
"""
liver ( label=1 ), kidney ( label=2 ), spleen ( label=3 ), and pancreas ( label=4 )
"""

def nii2slices(nii_path):
    image = sitk.ReadImage(nii_path)
    array = sitk.GetArrayFromImage(image)

    return array


def gen_train_slices():
    import re
    train_dir = "../data/FLARE21/TrainingImg"
    mask_dir = "../data/FLARE21/TrainingMask"

    train_samples = glob.glob(f'{train_dir}/*/*.nii.gz')

    train = []
    case_no = re.compile(r"\d{3,4}")
    save_dir = "../data/FLARE21/train_npz"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for f in train_samples:
        ct_slices = nii2slices(f)
        basename = os.path.basename(f)
        basename = '_'.join(basename.split('.')[0].split('_')[:-1])
        mask = os.path.join(mask_dir, basename)
        gt_slices = nii2slices(mask)
        assert len(ct_slices) == len(gt_slices)
        for i, (cslice, gslice) in enumerate(zip(ct_slices, gt_slices)):
            no = case_no.findall(f)[0]
            np.savez(os.path.join(save_dir, f"case{no}_slice{i:0>3}.npz"), image=cslice, label=gslice)
            train.append(f"case{no}_slice{i:0>3}\n")
        print(f"{f} with {i+1} slices are processed ...")

    list_dir = "../lists/lists_FLARE21"
    if not os.path.exists(list_dir):
        os.mkdir(list_dir)

    with open(os.path.join(list_dir, "train.txt"), 'w') as f:
        f.writelines(train)
    # print(train.__len__())
    print(f"slice generated in {list_dir}/train.txt")


def gen_test_data():
    import re
    import h5py
    test_dir = "../../medical/abdominal-multi-organ-segmentation/dataset/prep_val"
    ct_dir = os.path.join(test_dir, "CT")
    gt_dir = os.path.join(test_dir, "GT")

    files = os.listdir(ct_dir)
    test = []
    case_no = re.compile(r"\d{4}")
    save_dir = "../data/Synapse/test_vol_h5"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, f in enumerate(files):
        ct_slices = nii2slices(os.path.join(ct_dir, f))
        gt_slices = nii2slices(os.path.join(gt_dir, f.replace("img", "label")))
        assert len(ct_slices) == len(gt_slices)
        no = case_no.findall(f)[0]
        case = h5py.File(os.path.join(save_dir, f"case{no:0>4}.npy.h5"), 'w')
        case.create_dataset('image', data=ct_slices, shape=ct_slices.shape, dtype=ct_slices.dtype)
        case.create_dataset('label', data=gt_slices, shape=gt_slices.shape, dtype=gt_slices.dtype)
        image = sitk.ReadImage(os.path.join(ct_dir, f))
        case.create_dataset('spacing', data=image.GetSpacing())
        case.create_dataset('direction', data=image.GetDirection())
        case.create_dataset('origin', data=image.GetOrigin())
        case.flush()
        case.close()
        print(f"{f} case are processed ...")
        test.append(f"case{no:0>4}\n")

    with open('../lists/lists_Synapse/test_vol.txt', "w") as f:
        f.writelines(test)

    print(f"Cases generated in {'../lists/lists_Synapse/test_vol.txt'}")

    pass


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


class FLARE21_dataset(Dataset):
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


if __name__ == "__main__":
    # 生成训练数据集
    gen_train_slices()
    # 生成测试数据集
    # gen_test_data()
