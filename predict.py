import os
import re
import scipy
import numpy as np
import pytorch_lightning as pl
from predict_data_lightning import MyDataModule
from predict_model_lightning import MyLightningModel


if __name__ == '__main__':
    # == [参数] ================================================================================================ #
    # [修改-1: 路径dir]
    interp_type = ''
    add_name = ''   # 4eles_mon_, 4eles_day_, 3eles_mon_, 3eles_day_
    elements = ['target_sss', 'input_sss_last', 'input_ssh', 'input_sst', 'input_ssp', 'input_sse']
    # elements = ['target_sss', 'input_sss_last', 'input_ssh', 'input_sst', 'input_ssp', 'input_sse']
    # elements = ['target_sss', 'input_sss_first', 'input_ssh', 'input_sst', 'input_ssp', 'input_sse_clim_day']
    predict_total_samples = 366  # 总推理天数
    ckpt_dir = r"D:\Project\S4R2\model\New_from_20250530\Best_Model_GS\PRL_30_2025y_07m07d_20h35m_S4R2_M_unmask_None\best_epoch=33_val_mse_epoch=0.269466.ckpt"
    seed = 30
    region = "GS"  # "GS"  "KC"
    clim_mode = "keep"
    gyh_region = "File"
    gyh_type = "Norm"  # "Norm"

    # 自动矫正
    ckpt_dir = ckpt_dir.replace('\\', '/')
    data_dir = "D:/Project/S4R2/data/new/"
    # 使用正则表达式来提取字段
    result = None
    match = re.search(r'best_epoch=(\d+)_val_mse_epoch=(\d+\.\d+)', ckpt_dir)
    if match:
        best_epoch = match.group(1)
        val_mse_epoch = match.group(2)
        result = f"best_epoch={best_epoch}_val_mse_epoch={val_mse_epoch}"
        print(result)
    else:
        print("未找到匹配的字段")

    # [推理批量大小]
    predict_batch_size = 1
    predict_num_workers = 1

    # [生成路径]
    target_dir = os.path.split(ckpt_dir)[0].split("/")[-1]
    output_dir = f"D:/Project/S4R2/predict/New_from_20250530/{region}/{interp_type[1:]}{interp_type[0:1]}{add_name}{target_dir}_{result}/"
    params_dir = f"{output_dir}predict_changeable_data_params.txt"
    os.makedirs(output_dir, exist_ok=True)

    # [检查: 打印上述参数并检查]
    print(f"seed =  {seed}")
    print(f"ckpt_dir =  {ckpt_dir}")
    print(f"data_dir =  {data_dir}")
    print(f"output_dir =  {output_dir}")
    print(f"params_dir =  {params_dir}")
    print(f"region = {region}")
    print(f"clim_mode = {clim_mode}")
    print(f"gyh_region = {gyh_region}")
    print(f"gyh_type = {gyh_type}")
    print(f"elements = {elements}")
    print(f"predict_batch_size = {predict_batch_size}")
    print(f"predict_total_samples = {predict_total_samples}")
    print(f"predict_num_workers = {predict_num_workers}")

    # [备份: 把可变的data_params写入备份txt]
    with open(params_dir, "w") as file:
        file.write(f"seed =  {seed}\n")
        file.write(f"region = {region}\n")
        file.write(f"clim_mode = {clim_mode}\n")
        file.write(f"gyh_region = {gyh_region}\n")
        file.write(f"gyh_type = {gyh_type}\n")
        file.write(f"elements = {elements}\n")

    # == [实例化] ================================================================================================ #
    data_params = {
        # (0)基本参数
        "interp_type": interp_type,
        "data_dir": data_dir,
        "region": region,
        "clim_mode": clim_mode,

        "predict_batch_size": predict_batch_size,
        "predict_total_samples": predict_total_samples,
        "predict_num_workers": predict_num_workers,

        "elements": elements,
        "gyh_region": gyh_region,
        "gyh_type": gyh_type}  # 字典格式: 数据对象 = {归一化类型: {要素: [最小, 最大] 或 [平均, 方差]}}
    pl.seed_everything(seed, workers=True)  # 30/22:0.85  306:0.087  32:0.89
    data = MyDataModule(data_params)
    trainer = pl.Trainer(deterministic='warn',
                         accelerator="gpu",
                         devices=1)

    # == [推理] ================================================================================================ #
    model = MyLightningModel.load_from_checkpoint(ckpt_dir)  # 加载cpkt中的模型权重与偏置参数
    result = trainer.predict(model.double(), datamodule=data, return_predictions=True)
    # [提示] model 中包含 MyLightningModel的超参为: model_params 与 train_params
    # [传统] model = MyLightningModel(model_params, train_params)

    # == [保存] ================================================================================================ #
    input_origin = []
    input_minus = []
    output_origin = []
    output_minus = []
    target_origin = []
    target_minus = []

    for ele in ['output_origin', 'output_minus']:
        # 'input_origin', 'input_minus', 'output_origin', 'output_minus', 'target_origin', 'target_minus'
        for idx in range(predict_total_samples):
            if ele == 'input_origin':
                input_origin.append(result[-1][ele][idx].T)
            if ele == 'input_minus':
                input_minus.append(result[-1][ele][idx].T)
            if ele == 'output_origin':
                output_origin.append(result[-1][ele][idx].T)
            if ele == 'output_minus':
                output_minus.append(result[-1][ele][idx].T)
            if ele == 'target_origin':
                target_origin.append(result[-1][ele][idx].T)
            if ele == 'target_minus':
                target_minus.append(result[-1][ele][idx].T)

    # input_origin = np.squeeze(np.concatenate(input_origin, axis=3))
    # input_minus = np.squeeze(np.concatenate(input_minus, axis=3))
    output_origin = np.squeeze(np.concatenate(output_origin, axis=3))
    output_minus = np.squeeze(np.concatenate(output_minus, axis=3))
    # target_origin = np.squeeze(np.concatenate(target_origin, axis=3))
    # target_minus = np.squeeze(np.concatenate(target_minus, axis=3))

    # print(input_origin.shape)
    # print(input_minus.shape)
    print(output_origin.shape)
    print(output_minus.shape)
    # print(target_origin.shape)
    # print(target_minus.shape)

    # scipy.io.savemat(f'{output_dir}/input_origin.mat',
    #                  {'data': input_origin})
    # scipy.io.savemat(f'{output_dir}/input_minus.mat',
    #                  {'data': input_minus})
    # scipy.io.savemat(f'{output_dir}/target_origin.mat',
    #                  {'data': target_origin})
    # scipy.io.savemat(f'{output_dir}/target_minus.mat',
    #                  {'data': target_minus})
    scipy.io.savemat(f'{output_dir}/output_origin.mat',
                     {'data': output_origin})
    scipy.io.savemat(f'{output_dir}/output_minus.mat',
                     {'data': output_minus})
