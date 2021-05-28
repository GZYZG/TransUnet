# 使用多个视图进行inference

import torch
import argparse
from scipy.ndimage import zoom
import numpy as np
import os
import SimpleITK as sitk
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='FLARE21', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--is_chunked', type=bool, default=False, help="根据以往经验，截取序列中的指定范围进行分割")

args = parser.parse_args()


if __name__ == "__main__":
    model_path = {
        'X': '',
        'Y': '',
        'Z': ''
    }
    args.is_pretrain = True
    args.exp = 'TU_' + args.dataset + str(args.img_size)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    models = {}
    for view, path in model_path:
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        net.load_state_dict(torch.load(path))
        print(f"Load model from {path}")
        models[view] = net

    test_save_path = ''
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    patch_size = [args.img_size, args.img_size]
    test_dataset_path = './data/FLARE21/Validation/'
    niis = os.listdir(test_dataset_path)
    for nii in niis:
        nii_path = nii  # "3.nii.gz"
        image = sitk.ReadImage(os.path.join(test_dataset_path, nii_path))
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        image = sitk.GetArrayFromImage(image)

        if args.is_chunked:
            start_index = int(image.shape[0] * 0.40)  # 截取无效的切片
            image = image[start_index:]

        image = torch.from_numpy(image)
        image = image.squeeze(0).cpu().detach().numpy()

        predictions = {}
        shape = image.shape  # (z, x, y)
        shape = zip(['X', 'Y', 'Z'], shape)
        for idx, (view, dim) in enumerate(shape):
            predictions[view] = np.zeros_like(image)
            for ind in range(dim):
                if view == 'Z':
                    slice = image[ind, :, :]
                elif view == 'X':
                    slice = image[:, ind, :]
                else:
                    slice = image[:, :, ind]

                x, y = slice.shape[0], slice.shape[1]
                if x != patch_size[0] or y != patch_size[1]:
                    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

                models[view].eval()
                net = models[view]

                with torch.no_grad():
                    outputs = net(input)
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    if x != patch_size[0] or y != patch_size[1]:
                        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        pred = out

                    if view == 'Z':
                        predictions[view][ind, :, :] = pred
                    elif view == 'X':
                        predictions[view][:, ind, :] = pred
                    else:
                        predictions[view][:, :, ind] = pred

        # 根据三个视图的结果进行组装
        prediction = None

        if args.is_chunked:
            prediction = np.concatenate([np.zeros((start_index, *patch_size)), prediction], axis=0)
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing(spacing)
        prd_itk.SetOrigin(origin)
        prd_itk.SetDirection(direction)
        case = nii_path.replace(".nii.gz", "")

        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        print(f"{nii} processed finished...")
