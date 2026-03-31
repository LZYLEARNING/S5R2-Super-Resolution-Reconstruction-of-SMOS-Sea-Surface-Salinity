def data_statistics(file_path, clim_mode, region, gyh_type):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 定位关键词所在行
    key_lines = []
    for i, line in enumerate(lines):
        if clim_mode in line and region in lines[i+1]:
            key_lines.append(i + 2)

    statistics_dict = {}
    for line_index in key_lines:
        # 取出所在行的下一行的"Feature: "之后的字符串作为键K
        key_line = lines[line_index].strip().split(": ")[1]

        # 判断gyh_type类型，提取对应的统计数据
        if gyh_type == "Norm":
            mean_index = line_index + 1
            std_index = line_index + 2
            # 提取Mean和Std值并转换为数字
            mean_value = float(lines[mean_index].strip().split(": ")[1][:10])
            std_value = float(lines[std_index].strip().split(": ")[1][:10])
            statistics_dict[key_line] = [mean_value, std_value]
        elif gyh_type == "MinMax":
            min_index = line_index + 4
            max_index = line_index + 3
            # 提取Min和Max值并转换为数字
            min_value = float(lines[min_index].strip().split(": ")[1][:10])
            max_value = float(lines[max_index].strip().split(": ")[1][:10])
            statistics_dict[key_line] = [min_value, max_value]

    return statistics_dict


if __name__ == '__main__':
    # 示例用法
    file_path = 'D:/Project/S4R2/data/new/mr_statistics/statistics.txt'
    clim_mode = "keep"
    region = "KC"
    gyh_type = "MinMax"

    result = data_statistics(file_path, clim_mode, region, gyh_type)
    print(result)

    a = [34.0895062, 0.68656169]
    b = [34.0895062, 0.68656169]
    print(a+b)
