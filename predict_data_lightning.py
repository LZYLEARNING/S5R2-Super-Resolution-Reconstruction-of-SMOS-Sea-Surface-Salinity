import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_set import MyDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, kwargs):
        super().__init__()
        self.interp_type = kwargs.get('interp_type', '')
        self.data_dir = kwargs.get('data_dir', 'D:/Project/S4R2/data/new')

        self.predict_dataset = None
        self.predict_batch_size = kwargs.get('predict_batch_size', 1)
        self.predict_total_samples = kwargs.get('predict_total_samples', 366)
        self.predict_num_workers = kwargs.get('predict_num_workers', 1)

        self.gyh_region = kwargs.get('gyh_region', "File")
        self.region = kwargs.get('region', 'KC')
        self.elements = kwargs.get("elements")
        self.clim_mode = kwargs.get('clim_mode', 'keep')  # or "minus"

    def setup(self, stage=None):
        predict_data_path = self.data_dir + '/mr_test' + self.interp_type
        print("predict_data_path", predict_data_path)

        self.predict_dataset = MyDataset(gyh_region=self.gyh_region,
                                         clim_mode=self.clim_mode,
                                         data_path=predict_data_path,
                                         region=self.region,
                                         elements=self.elements)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.predict_batch_size,
                          shuffle=False, num_workers=self.predict_num_workers, persistent_workers=True, pin_memory=True)

