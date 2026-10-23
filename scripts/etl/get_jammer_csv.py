import os
import pandas as pd
import json

# 设置文件夹路径
folder_path = "./data"
json_path = "./json"

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        jammer_type = []
        jammer = []
        radar = []
        SINR = []
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            df = df.iloc[:-1]
            # 遍历每一列
            for column in df.columns:
                column_data = df[column].tolist()
                
                # 构造JSON文件名
                base_name = os.path.splitext(filename)[0]
                json_filename = f"{base_name}_{column}.json"
                json_path = os.path.join(folder_path, json_filename)

                # 保存JSON文件
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(column_data, json_file, indent=4)

                print(f"Saved: {json_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
