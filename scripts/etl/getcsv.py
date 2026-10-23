import os
import pandas as pd
import json

# 设置文件夹路径
folder_path = "./data"
output_json_folder = "./json"
import ast

def parse_tuple_string(s):
    try:
        # 安全地将字符串转换为元组
        parsed = ast.literal_eval(s)
        if not isinstance(parsed, (tuple, list)):
            raise ValueError("Input is not a tuple or list.")
        
        # 转换 True/False 为 1/0
        result = [int(x) if isinstance(x, bool) else x for x in parsed]
        return result
    
    except (ValueError, SyntaxError) as e:
        print(f"无法解析字符串：{s}，错误：{e}")
        return []


# 确保输出文件夹存在
os.makedirs(output_json_folder, exist_ok=True)

# 用于存储每一列的合并数据
combined_columns = {}

# 遍历所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        try:
            # 读取 CSV 并去除最后一行
            df = pd.read_csv(file_path).iloc[:-1]
            df = df.iloc[:-1]

            # 遍历所有列，将数据追加到对应列表中
            for column in df.columns:
                if column not in combined_columns:
                    combined_columns[column] = []
                combined_columns[column].extend(df[column].tolist())

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 保存每一列合并后的数据为单独的 JSON 文件
for column, data in combined_columns.items():
    json_filename = f"{column}.json"
    json_path = os.path.join(output_json_folder, json_filename)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Saved combined column '{column}' to: {json_path}")


new_data = []
with open("./json/jammer.json", "r", encoding="utf-8") as f:
    jamming_data = json.load(f)
    for data in jamming_data:
        parsed = parse_tuple_string(data)
        new_data.append(parsed)


with open("./json/jammer.json", "w", encoding="utf-8") as json_file:
    json.dump(new_data, json_file, indent=4)