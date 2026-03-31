import random
import torch
import scipy.io
from scipy.io import savemat


def data_transform(datas, stcs, gyh_region, gyh_type, elements, p_crop, p1, p2, p3, p4, p5):
    lr_miss_mask = None

    for i in range(len(datas) - 1, -1, -1):
        sample = datas[i]

        # [数据增强]
        # (1) 随机裁剪
        H, W = sample.size()[1], sample.size()[2]
        if i == len(datas) - 1:  # len(datas) - 1 就是列表中的最后一个！
            while True:
                # 计算裁剪后的尺寸
                target_h = int(H * p_crop)
                target_w = int(W * p_crop)

                # 生成随机位置
                top = int(p1 * (H - target_h))
                left = int(p1 * (W - target_w))

                # H 和 W 的索引都是从 0 开始的，即最顶部像素和最左侧像素的索引为 0。
                # 裁剪图片
                sample_crop = sample[:, top:top + target_h, left:left + target_w].contiguous()
                one_percentage = torch.sum(sample_crop[0] == 1).item() / (target_h * target_w)
                if one_percentage > 0.50:
                    break
                else:
                    p1 = random.random()
        else:
            # 计算裁剪后的尺寸
            target_h = int(H * p_crop)
            target_w = int(W * p_crop)
            # 生成随机位置
            top = int(p1 * (H - target_h))
            left = int(p1 * (W - target_w))
            # 裁剪图片
            sample_crop = sample[:, top:top + target_h, left:left + target_w].contiguous()

        # (2) 水平翻转
        if p2 > 0.5:
            # 随机水平翻转
            sample_crop = sample_crop.flip(-1)

        # (3) 垂直翻转
        if p3 > 0.5:
            # 随机垂直翻转
            sample_crop = sample_crop.flip(-2)

        # (4) 旋转
        if p4 > 0.5:
            # 随机旋转180度
            sample_crop = torch.flip(sample_crop, dims=(-2, -1))

        # (5) Local Augment
        if p5 > 0.5:
            h, w = sample_crop.shape[-2:]
            s1 = sample_crop[:, :h // 2, :w // 2].contiguous()
            s2 = sample_crop[:, h // 2:, :w // 2].contiguous()
            s3 = sample_crop[:, :h // 2, w // 2:].contiguous()
            s4 = sample_crop[:, h // 2:, w // 2:].contiguous()

            s1 = torch.flip(s1, dims=(-2, -1))
            s2 = torch.flip(s2, dims=(-2, -1))
            s3 = torch.flip(s3, dims=(-2, -1))
            s4 = torch.flip(s4, dims=(-2, -1))

            sample_crop = torch.cat([torch.cat([s1, s2], dim=-2), torch.cat([s3, s4], dim=-2)], dim=-1)

        # [数据归一化]
        if gyh_region == 'Point':
            if i < len(elements):
                stc = stcs[elements[i]]
                if gyh_type == 'MinMax':
                    sample_crop = (sample_crop - stc[0]) / (stc[1] - stc[0])
                if gyh_type == 'Norm':
                    sample_crop = (sample_crop - stc[0]) / stc[1]

        if gyh_region == 'File':
            if i < len(elements):
                if gyh_type == 'MinMax':
                    max_mat = datas[len(elements) + 4 * i + 2].squeeze(dim=1).contiguous()
                    min_mat = datas[len(elements) + 4 * i + 3].squeeze(dim=1).contiguous()
                    sample_crop = (sample_crop - min_mat) / (max_mat - min_mat)
                if gyh_type == 'Norm':
                    avg_mat = datas[len(elements) + 4 * i].squeeze(dim=1).contiguous()
                    std_mat = datas[len(elements) + 4 * i + 1].squeeze(dim=1).contiguous()
                    sample_crop = (sample_crop - avg_mat) / std_mat

        # [获取缺失值掩膜]
        if i == 1:
            lr_miss_mask = torch.where(torch.isnan(sample_crop), torch.zeros_like(sample_crop), torch.ones_like(sample_crop)).unsqueeze(1).contiguous()

        # [数据去除nan]
        sample_crop[sample_crop != sample_crop] = torch.tensor(0., dtype=sample_crop.dtype)

        datas[i] = sample_crop.unsqueeze(1).contiguous()

    if gyh_region == 'File':
        return (datas[0],
                datas[1:len(elements)],
                datas[len(elements): 5 * len(elements)],
                datas[5 * len(elements): 5 * len(elements) + 2],
                datas[5 * len(elements) + 2:],
                lr_miss_mask)

    if gyh_region == 'Point':
        return (datas[0],
                datas[1:len(elements)],
                datas[len(elements):len(elements) + 2],
                datas[len(elements) + 2:],
                lr_miss_mask)
