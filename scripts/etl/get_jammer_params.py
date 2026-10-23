import json

# 加载原始数据
with open('data/v2/json\jammer.json', 'r') as f:
    data = json.load(f)

# 处理数据：删除每个子列表中的前两个元素，并将每个子列表的维度补充到10个
new_data = []
for item in data:
    # 删除前两个元素
    new_item = item[2:]
    
    # 如果新元素的维度小于10，使用0进行补充
    while len(new_item) < 10:
        new_item.append(0)
    
    # 对字符串进行从0开始的编码（如果字符串出现在数据中）
    new_item_encoded = []
    for elem in new_item:
        if isinstance(elem, str):  # 如果元素是字符串类型
            # 创建一个简单的编码器（这里将每个唯一的字符串映射为0，1，2等）
            if elem not in new_item_encoded:
                new_item_encoded.append(elem)
        else:
            new_item_encoded.append(elem)

    # 重新编码字符串部分（映射为从0开始的数字）
    new_item_encoded = [new_item_encoded.index(elem) if isinstance(elem, str) else elem for elem in new_item]

    new_data.append(new_item_encoded)

# 保存新的数据到一个JSON文件
with open('data/v2/json\jammer_params.json', 'w') as f:
    json.dump(new_data, f)

