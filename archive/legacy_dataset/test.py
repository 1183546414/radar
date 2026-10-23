import pandas as pd
import re


def extract_values(line, string_encoding_map):
    values = []
    
    # 提取 key-value 对（包括带单位、布尔、字符串）
    pairs = re.findall(r'(\w+):\s*([^,\]\)]+)', line)
    
    for key, val in pairs:
        val = val.strip()
        # 处理布尔值
        if val.lower() == 'true':
            values.append(1)
        elif val.lower() == 'false':
            values.append(0)
        # 处理数字（支持小数、科学记数法）
        elif re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', val):
            if '.' in val or 'e' in val.lower():
                values.append(float(val))
            else:
                values.append(int(val))
        # 处理带单位的数字（如 '1000 ns'）
        elif re.match(r'^-?\d+(\.\d+)?(?:[eE][-+]?\d+)?\s*\w+$', val):
            num = re.match(r'^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', val).group(1)
            values.append(float(num) if '.' in num or 'e' in num.lower() else int(num))
        # 处理字符串值（进行编码）
        else:
            if val not in string_encoding_map:
                string_encoding_map[val] = len(string_encoding_map)
            values.append(string_encoding_map[val])
    
    # 处理像 (1, 2, 3) 这样的元组
    if line.strip().startswith("(") and line.strip().endswith(")"):
        numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
        for n in numbers:
            values.append(float(n) if '.' in n else int(n))

    return values


# 示例数据：每行是一个字符串


# 用于字符串编码的全局映射表
string_encoding_map = {}

df = pd.read_csv('jamming_radar_RANDOM_sinr_data_2025-04-28_19-55-55.csv', encoding='gbk')
s2 = df.iloc[53,2]

# 逐行处理
all_results = extract_values(s2, string_encoding_map)

# 输出
print(all_results)

# 显示字符串编码映射
print("\nString Encoding Map:", string_encoding_map)
