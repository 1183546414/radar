import json
def parse_json_to_arrays(file_path):
    with open(file_path, 'r') as file:
        # 加载 JSON 数据
        data = json.load(file)
        
        # 创建一个空列表，用来保存转换后的数组
        arrays = []
        for item in data:
            # 使用 eval() 将每个字符串转换为列表
            arrays.append(eval(item))  # eval() 将字符串表示的列表转换成实际的列表
        
        return arrays


def save_arrays_to_files(arrays, file_name):

    # 打开一个新的 JSON 文件并写入该数组
    with open(file_name, 'w') as file:
            json.dump(arrays, file)
            
file_path = './json/radar.json'  # JSON 文件的路径
new_file_path = "./json/radar_trans.json"
arrays = parse_json_to_arrays(file_path)  # 解析 JSON 文件中的数组
save_arrays_to_files(arrays, new_file_path)  # 将每个数组保存到不同的文件中
