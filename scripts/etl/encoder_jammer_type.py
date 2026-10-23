import json

# 读取原始数组
with open("data/v2/json\jammer_type.json", "r", encoding="utf-8") as f:
    arr = json.load(f)

# 编码映射表
mapping = {}
encoded_arr = []
current_code = 0

for item in arr:
    if item not in mapping:
        mapping[item] = current_code
        current_code += 1
    encoded_arr.append(mapping[item])

print(mapping)
# 保存编码后的数组到 JSON 文件
with open("data/v2/json\jammer_type_encoded.json", "w", encoding="utf-8") as f:
    json.dump(encoded_arr, f, ensure_ascii=False, indent=4)

print("编码完成，结果已保存到 jammer_type_encoded.json")
