import os
import h5py
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,
                 gyh_region,
                 clim_mode,
                 data_path,
                 region,
                 elements):

        # [读取气候态数据clim]
        self.gyh_region = gyh_region
        self.clim_mode = clim_mode
        self.clims_dir = [os.path.join(os.path.dirname(data_path) + '/mr_clims' + '/' + region, feature)
                          for feature in elements]

        self.statistics_dir = [os.path.join(os.path.dirname(data_path) + '/mr_statistics' + '/' + region, feature)
                               for feature in elements]

        # [读取原始数据origins]
        self.elements_dir = [os.path.join(data_path + '/origins' + '/' + region, feature)
                             for feature in elements]
        self.masks_dir = [os.path.join(data_path + '/masks/standard_025degree_mask/'
                                       + region + "_standard_025degree_mask.mat"),
                          os.path.join(data_path + '/masks/standard_0083degree_mask/'
                                       + region + "_standard_0083degree_mask.mat")]

        self.file_list = self._get_file_list(self.elements_dir[0])
        self.region = region

    def _get_file_list(self, path):
        files = os.listdir(path)
        file_list = []
        for file_name in files:
            if file_name.endswith('.mat'):
                file_path = os.path.join(path, file_name)
                file_list.append(file_path)
        print(file_list)
        print(len(file_list))
        return file_list

    def _load_hdf(self, file_path):
        with h5py.File(file_path, 'r') as f:
            data = f.get(list(f.keys())[0])[:]
        return data

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # [批次]
        batch = []

        # [文件名]
        element_name = os.path.basename(self.file_list[index])
        clim_name = element_name[4:6]

        # [输入]
        for element_dir, clim_dir in zip(self.elements_dir, self.clims_dir):
            element = self._load_hdf(os.path.join(element_dir, element_name))

            if self.clim_mode == "minus":
                clim = self._load_hdf(os.path.join(clim_dir, clim_name + '.mat'))
                element = np.subtract(element, clim)

            batch.append(element)

        # [统计量mat]
        if self.gyh_region == 'File':
            for statistic_dir in self.statistics_dir:
                statistic = self._load_hdf(os.path.join(statistic_dir, "statistics_"+self.clim_mode+".mat"))
                avg_mat = statistic[0]
                std_mat = statistic[1]
                max_mat = statistic[2]
                min_mat = statistic[3]

                batch.append(avg_mat)
                batch.append(std_mat)
                batch.append(max_mat)
                batch.append(min_mat)

        clim_input_sss = self._load_hdf(os.path.join(self.clims_dir[1], clim_name + '.mat'))
        clim_target_sss = self._load_hdf(os.path.join(self.clims_dir[0], clim_name + '.mat'))

        batch.append(clim_input_sss)
        batch.append(clim_target_sss)

        # [掩膜]
        batch.append(self._load_hdf(self.masks_dir[0]))
        batch.append(self._load_hdf(self.masks_dir[1]))

        return batch


# ######################################################################################################################
# data = MyDataset(data_dir)说明
# 根据给出的代码和输出，我们无法确定数据集是否已经完全加载到了程序中。
# 输出只显示了 `data` 的字符串表示形式，没有提供关于数据是否加载的信息。
# 通常情况下，当你创建一个数据集对象时，并不会立即将所有数据加载到内存中。
# 相反，数据集对象通常会维护一个指向数据的引用或路径，并在需要时逐批加载数据。这样可以节省内存空间并提高效率。
# 如果你担心数据集占用过多的内存，可以查看 `MyDataset` 类的实现代码，看看是否有明确的数据加载机制。
# 或者，你可以尝试使用数据集对象的方法（如 `len()`）来获取数据集的大小，以便评估其占用的内存空间。
# 另外，如果你的数据集非常庞大，超过了系统的可用内存大小，那么一次性加载整个数据集可能导致内存溢出。
# 在这种情况下，你可以考虑使用分批次加载数据或者使用更高级的数据加载策略来处理数据。
# 总之，只有指定了dataset后的索引[]，才可以实现加载数据，不会一次性把所有数据加载进来，占用内存，NICE！
# ######################################################################################################################
