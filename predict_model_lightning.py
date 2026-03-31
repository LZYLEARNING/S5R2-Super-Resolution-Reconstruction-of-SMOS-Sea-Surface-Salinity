import torch
import pytorch_lightning as pl
from data_transform import data_transform


class MyLightningModel(pl.LightningModule):
    def __init__(self, model_params, train_params):
        super(MyLightningModel, self).__init__()
        self.all_predictions = {
            'input_origin': [],
            'input_minus': [],
            'output_origin': [],
            'output_minus': [],
            'target_origin': [],
            'target_minus': []
        }

        self.save_hyperparameters()

        self.model_mode = train_params.get('model_mode', 'S4R2_MH_mask')

        # [OSM = Ocean Specific Mask = mask]
        if self.model_mode == 'S4R2_MH_mask':
            from model_S4R2_MH_mask import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_HM_mask':
            from model_S4R2_HM_mask import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_M_mask':
            from model_S4R2_MH_mask import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_M_mask_NoGAM':
            from model_S4R2_MH_mask_NoGAM import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_M_mask_NoMFB':
            from model_S4R2_MH_mask_NoMFB import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_H_mask':
            from model_S4R2_HM_mask import S4R2
            self.model = S4R2(model_params)

        # [LMO = Land Mixed Ocean = unmask]
        elif self.model_mode == 'S4R2_MH_unmask':
            from model_S4R2_MH_unmask import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_HM_unmask':
            from model_S4R2_HM_unmask import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_M_unmask':
            from model_S4R2_MH_unmask import S4R2
            self.model = S4R2(model_params)
        elif self.model_mode == 'S4R2_H_unmask':
            from model_S4R2_HM_unmask import S4R2
            self.model = S4R2(model_params)

        # [CNN series model]
        elif self.model_mode == 'SRCNN':
            from model_SRCNN import SRCNN
            self.model = SRCNN(model_params)
        elif self.model_mode == 'VDSR':
            from model_VDSR import VDSR
            self.model = VDSR(model_params)
        elif self.model_mode == 'SRDN':
            from model_SRDN import SRDN
            self.model = SRDN(model_params)
        elif self.model_mode == 'EDSR':
            from model_EDSR import EDSR
            self.model = EDSR(model_params)

        # [Transformer-based series model]
        # 2021
        elif self.model_mode == 'SwinIR':
            from model_SwinIR import SwinIR
            self.model = SwinIR(model_params)
        elif self.model_mode == 'SwinIR_light':
            from model_SwinIR_light import SwinIR_light
            self.model = SwinIR_light(model_params)

        # 2022
        elif self.model_mode == 'ESRT':
            from model_ESRT import ESRT
            self.model = ESRT(model_params)

        # 2023
        elif self.model_mode == 'CRAFT':
            from model_CRAFT import CRAFT
            self.model = CRAFT(model_params)
        elif self.model_mode == 'CRAFT2':
            from model_CRAFT2 import CRAFT2
            self.model = CRAFT2(model_params)

        # 2024
        elif self.model_mode == 'RGT':
            from model_RGT import RGT
            self.model = RGT(model_params)
        elif self.model_mode == 'RGT_light':
            from model_RGT_light import RGT_light
            self.model = RGT_light(model_params)

        # 2023 & 2024
        elif self.model_mode == 'SRFormer':
            from model_SRFormer import SRFormer
            self.model = SRFormer(model_params)
        elif self.model_mode == 'SRFormer_light':
            from model_SRFormer_light import SRFormer_light
            self.model = SRFormer_light(model_params)

        # GAN
        elif self.model_mode == 'SRGAN':
            from model_SRGAN_G import MSRResNet
            self.model_g = MSRResNet(model_params)
            from model_SRGAN_D import VGGStyleDiscriminator
            self.model_d = VGGStyleDiscriminator(model_params)
        elif self.model_mode == 'ESRGAN':
            from model_ESRGAN_G import RRDBNet
            self.model_g = RRDBNet(model_params)
            from model_SRGAN_D import VGGStyleDiscriminator
            self.model_d = VGGStyleDiscriminator(model_params)

        print("self.model_mode", self.model_mode)

        self.clim_mode = "keep"  # train_params.get('clim_mode', 'keep')

        self.elements = train_params.get('elements', )
        self.stcs = train_params.get('stcs')
        self.target_sss_norm = train_params.get('target_sss_norm', [30.62074645422399, 4.875331884250045])
        self.target_sss_minmax = train_params.get('target_sss_minmax', [30.62074645422399, 4.875331884250045])
        self.input_sss_norm = train_params.get('input_sss_norm', [30.62074645422399, 4.875331884250045])
        self.input_sss_minmax = train_params.get('input_sss_minmax', [30.62074645422399, 4.875331884250045])

        self.gyh_region = train_params.get('gyh_region', "Point")  # 'Point's
        self.gyh_type = train_params.get('gyh_type', "MinMax")  # [min, diff]

    def forward(self, inputs, masks):
        if "S4R2" in self.model_mode:
            return self.model(inputs, masks)
        elif (self.model_mode == "SRCNN"
              or self.model_mode == "VDSR"
              or self.model_mode == "SRDN"
              or self.model_mode == "EDSR"
              or self.model_mode == "SwinIR"
              or self.model_mode == "SwinIR_light"
              or self.model_mode == "ESRT"
              or self.model_mode == "CRAFT"
              or self.model_mode == "CRAFT2"
              or self.model_mode == "RGT"
              or self.model_mode == "RGT_light"
              or self.model_mode == "SRFormer"
              or self.model_mode == "SRFormer_light"):
            return self.model(inputs, masks)
        elif (self.model_mode == "SRGAN"
              or self.model_mode == "ESRGAN"):
            return self.model_g(inputs, masks)

    def predict_step(self, batch):
        self.eval()
        target = None
        inputs = None
        statistics = None
        clims = None
        masks = None
        lr_miss_mask = None
        input_origin = None
        input_minus = None
        output_origin = None
        output_minus = None
        target_origin = None
        target_minus = None

        if self.gyh_region == 'Point':
            target, inputs, clims, masks, lr_miss_mask = data_transform(batch,
                                                                        self.stcs,
                                                                        self.gyh_region,
                                                                        self.gyh_type,
                                                                        self.elements,
                                                                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if self.gyh_region == 'File':
            target, inputs, statistics, clims, masks, lr_miss_mask = data_transform(batch,
                                                                                    self.stcs,
                                                                                    self.gyh_region,
                                                                                    self.gyh_type,
                                                                                    self.elements,
                                                                                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # input = inputs[0]
        output = self(inputs, masks)  # 相当于运行一个forward

        # [掩膜规整]
        hr_mask = masks[1]

        if self.gyh_region == 'Point':
            if self.gyh_type == "Norm":
                # input = ((input * self.input_sss_norm[1]) + self.input_sss_norm[0])
                # input[lr_miss_mask == 0] = float('nan')

                output = ((output * self.target_sss_norm[1]) + self.target_sss_norm[0])
                output[hr_mask == 0] = float('nan')

                # target = ((target * self.target_sss_norm[1]) + self.target_sss_norm[0])
                # target[hr_mask == 0] = float('nan')

            if self.gyh_type == "MinMax":
                # input = ((input * (self.input_sss_minmax[1] - self.input_sss_minmax[0])) + self.input_sss_minmax[0])
                # input[lr_miss_mask == 0] = float('nan')

                output = ((output * (self.target_sss_minmax[1] - self.target_sss_minmax[0])) + self.target_sss_minmax[0])
                output[hr_mask == 0] = float('nan')

                # target = ((target * (self.target_sss_minmax[1] - self.target_sss_minmax[0])) + self.target_sss_minmax[0])
                # target[hr_mask == 0] = float('nan')

        elif self.gyh_region == 'File':
            # input_avg_mat = statistics[4]
            # input_std_mat = statistics[5]
            # input_max_mat = statistics[6]
            # input_min_mat = statistics[7]

            target_avg_mat = statistics[0]
            target_std_mat = statistics[1]
            target_max_mat = statistics[2]
            target_min_mat = statistics[3]

            if self.gyh_type == "Norm":
                # input = input * input_std_mat + input_avg_mat
                # input[lr_miss_mask == 0] = float('nan')

                output = output * target_std_mat + target_avg_mat
                output[hr_mask == 0] = float('nan')

                # target = target * target_std_mat + target_avg_mat
                # target[hr_mask == 0] = float('nan')

            elif self.gyh_type == "MinMax":
                # input = input * (input_max_mat - input_min_mat) + input_min_mat
                # input[lr_miss_mask == 0] = float('nan')

                output = output * (target_max_mat - target_min_mat) + target_min_mat
                output[hr_mask == 0] = float('nan')

                # target = target * (target_max_mat - target_min_mat) + target_min_mat
                # target[hr_mask == 0] = float('nan')

        # [保存原态origin与气候态minus]
        if self.clim_mode == 'keep':
            # input_origin = input
            # input_minus = torch.subtract(input_origin, clims[0])

            output_origin = output
            output_minus = torch.subtract(output_origin, clims[1])

            # target_origin = target
            # target_minus = torch.subtract(target_origin, clims[1])

        elif self.clim_mode == 'minus':
            # input_minus = input
            # input_origin = torch.add(input_minus, clims[0])

            output_minus = output
            output_origin = torch.add(output_minus, clims[1])

            # target_minus = target
            # target_origin = torch.add(target_minus, clims[1])

        # 将预测结果添加到类变量中
        # self.all_predictions['input_origin'].append(input_origin.cpu().detach().numpy())
        # self.all_predictions['input_minus'].append(input_minus.cpu().detach().numpy())
        self.all_predictions['output_origin'].append(output_origin.cpu().detach().numpy())
        self.all_predictions['output_minus'].append(output_minus.cpu().detach().numpy())
        # self.all_predictions['target_origin'].append(target_origin.cpu().detach().numpy())
        # self.all_predictions['target_minus'].append(target_minus.cpu().detach().numpy())

        return self.all_predictions
