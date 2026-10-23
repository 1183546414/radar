import json

# 加载原始数据
with open('data/v2/json\jammer.json', 'r') as f:
    data = json.load(f)

# 提取每个子列表中的前两个元素
new_data = [[item[0], item[1]] for item in data]

# 将新数据保存到一个JSON文件
with open('data/v2/json\jammer_carrier.json', 'w') as f:
    json.dump(new_data, f)
