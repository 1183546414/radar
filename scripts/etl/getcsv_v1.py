import pandas as pd
import re
import json

#雷达数据的数据类
class Rdata:
    Jamming_Type = ''
    Jammer_carrierFrequency = 0
    Jammer_gain = 0

    Time = 0
    Sleep = 0
    CarrierFreq = 0
    BandWidth = 0
    NoiseType = 0
    Depth = 0
    AutoAim = 0

    Delay = 0
    Number = 0
    IntervalMin = 0
    IntervalMax = 0
    DoppleShift = 0

    Type = ''
    SamplingTime = 0
    DelayTime = 0
    IntervalTime = 0
    RepeatNumber = 0
    CycleNumber = 0

    # Time = 0
    # Sleep = 0
    PWMax = 0
    PWMin = 0
    PowerMax = 0
    PowerMin = 0
    PRFMax = 0
    PRFMin = 0
    CarrierFreq = 0
    # Bandwidth = 0

    Radar_carrierFrequency = 0
    Radar_gain = 0
    Radar_type = '' 
    Radar_bandwidth = 0
    pulseWidth = 0
    chirpType = ''

    rootIndex = 0

    startFrequency=0
    stepFrequency = 0
    stepNumber = 0
    stepDuration = 0

    SINR = 0

#获取雷达数据的Jammer_carrierFrequency
def get_Jammer_carrierFrequency(s):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    numbers = [float(num) if '.' in num else int(num) for num in numbers]
    # 分别存为变量
    result = {}
    carrier_frequency = numbers[0]
    gain = numbers[1]
    
    return [carrier_frequency, gain]


#获取雷达数据的Jamming_Params（版本1）
def get_Jamming_Params(s):
    # 使用正则表达式提取键值对：Key: Value
    pattern = r"(\w+): ([\d\.]+)"
    #pattern = r"(\w+):\s*([^,\]]+)"
    
    matches = re.findall(pattern, s)

    # 将结果转换为字典，数字统一转为 float 或 int
    result = {}
    for key, value in matches:
        # 尝试将值转换为数字
        if '.' in value:
            result[key] = float(value)
        else:
            result[key] = int(value)

    # 若某些键缺失，赋默认值
    result.setdefault("Time", 0)
    result.setdefault("Sleep", 0)
    result.setdefault("CarrierFreq", 0)
    result.setdefault("Bandwidth", 0)
    result.setdefault("NoiseType", 0)
    result.setdefault("Depth", 0) 

    result.setdefault("Delay", 0)
    result.setdefault("Number", 0)    
    result.setdefault("IntervalMin", 0)  
    result.setdefault("IntervalMax", 0)  
    result.setdefault("DoppleShift", 0)  

    #result.setdefault("Type", ' ')  
    result.setdefault("SamplingTime", 0)  
    result.setdefault("DelayTime", 0)  
    result.setdefault("IntervalTime", 0)  
    result.setdefault("RepeatNumber", 0)  
    result.setdefault("CycleNumber", 0)  

    result.setdefault("PWMax", 0)  
    result.setdefault("PWMin", 0)  
    result.setdefault("PowerMax", 0)  
    result.setdefault("PowerMin", 0)  
    result.setdefault("PRFMax", 0)  
    result.setdefault("PRFMin", 0)  
    result.setdefault("CarrierFreq", 0)  
    result["Type"] = ''
    result["AutoAim"] = True

    # 单独处理布尔值
    if "AutoAim" in s:
        result["AutoAim"] = "AutoAim: True" in s

    if "Type" in s:
        type_match = re.search(r"Type:\s*(\w+)", s)
        type_value = type_match.group(1) if type_match else None
        result["Type"] = type_value
    return result



#获取雷达数据的Jamming_Params（版本1）
def get_Jamming_Params_2(line, string_encoding_map):
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



#获取雷达数据的Radar_carrierFrequency
def get_Radar_carrierFrequency(s):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    numbers = [float(num) if '.' in num else int(num) for num in numbers]
    # 分别存为变量
    result = {}
    carrier_frequency = numbers[0]
    gain = numbers[1]
    return [carrier_frequency,gain]


#获取雷达数据的Radar_Params
def get_Radar_Params(s):
    # 使用正则表达式提取键值对：Key: Value
    pattern = r"(\w+): ([\d\.]+)"
    #pattern = r"(\w+):\s*([^,\]]+)"
    
    matches = re.findall(pattern, s)

    # 将结果转换为字典，数字统一转为 float 或 int
    result = {}
    for key, value in matches:
        # 尝试将值转换为数字
        if '.' in value:
            result[key] = float(value)
        else:
            result[key] = int(value)
    
    result.setdefault("bandwidth", 0)
    result.setdefault("pulseWidth", 0)
    result.setdefault("CarrierFreq", 0)
    result.setdefault("rootIndex", 0)

    result.setdefault("startFrequency", 0)
    result.setdefault("stepFrequency", 0)
    result.setdefault("stepNumber", 0)
    result.setdefault("stepDuration", 0)
    result.setdefault("radar_type", '')
    result.setdefault("chirpType", '')

    if "radar_type" in s:
        type_match = re.search(r"radar_type:\s*(\w+)", s)
        type_value = type_match.group(1) if type_match else None
        result["radar_type"] = type_value

    if "chirpType" in s:
        type_match = re.search(r"chirpType:\s*(\w+)", s)
        type_value = type_match.group(1) if type_match else None
        result["chirpType"] = type_value
    return result


#获取雷达数据的SINR
def get_SINR(s):
    # 提取第一个浮点数或整数
    result = {}
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if match:
        value = float(match.group())
    return value


#对雷达数据的Jammer_carrierFrequency进行归一化（除以所有数据中的最大值）
def Jammer_carrierFrequency_normalization(Jammer_carrierFrequency):
    max1 = 0
    max2 = 0
    for data in Jammer_carrierFrequency:
        if data['Jammer_carrierFrequency'] > max1:
            max1 = data['Jammer_carrierFrequency']
        if data['gain'] > max2:
            max2 = data['gain']
        
    for data in Jammer_carrierFrequency:
        data['Jammer_carrierFrequency_normalization'] = data['Jammer_carrierFrequency'] / max1
        data['gain_normalization'] = data['gain'] / max2
    return Jammer_carrierFrequency


#对雷达数据的Jamming_Params进行归一化（除以所有数据中的最大值）
def Jamming_Params_normalization(Jamming_Params):
    params_list = ['Time','Sleep','CarrierFreq','Bandwidth','NoiseType','Depth','Delay','Number','IntervalMin','IntervalMax','DoppleShift',
                   'SamplingTime','DelayTime','IntervalTime', 'RepeatNumber','CycleNumber','PWMax','PWMin','PowerMax','PowerMin','PRFMax','PRFMin']
    for param in params_list:
        max = getMax(param, Jamming_Params)
        if max != 0:
            for data in Jamming_Params:
                data[param] = data[param] / max
    return Jamming_Params


#对于Jamming_Params，获取一个维度上的最大值
def getMax(param, Jamming_Params):
    list = []
    for data in Jamming_Params:
        print(data)
        list.append(data[param])
    max_value = max(list)
    return max_value



def pad_to_length_10(lst):
    """
    如果列表长度小于10，则用0补齐到长度10；如果大于等于10，则保持不变。
    """
    return lst + [0] * (10 - len(lst)) if len(lst) < 10 else lst

def pad_to_length_5(lst):
    """
    如果列表长度小于10，则用0补齐到长度10；如果大于等于10，则保持不变。
    """
    return lst + [0] * (5 - len(lst)) if len(lst) < 5 else lst



# 尝试使用 GBK 编码读取
df = pd.read_csv('jamming_radar_RANDOM_sinr_data_2025-04-28_19-55-55.csv', encoding='gbk')

Jamming_Type = []
Jammer_carrierFrequency = []
Jamming_Params = []
Radar_carrierFrequency = []
Radar_Params = []
SINR = []
Jamming_Type_dict = {}
Jamming_encoding_map = {}
Radar_encoding_map = {}



#开始读取雷达数据
for i in range(0, len(df) - 1):
#for i in range(0, 50):
    s0 = df.iloc[i,0]
    Jamming_Type.append({"Jamming_Type": s0})

    s1 = df.iloc[i,1]
    Jammer_carrierFrequency.append(get_Jammer_carrierFrequency(s1))

    s2 = df.iloc[i,2]
    param = get_Jamming_Params_2(s2, Jamming_encoding_map)
    Jamming_Params.append(pad_to_length_10(param))

    s3 = df.iloc[i,3]
    Radar_carrierFrequency.append(get_Radar_carrierFrequency(s3))

    s4 = df.iloc[i,4]
    param = get_Jamming_Params_2(s4, Radar_encoding_map)
    Radar_Params.append(pad_to_length_5(param))

    s5 = df.iloc[i,5]
    SINR.append(get_SINR(s5))

print(Jamming_encoding_map)
print(Radar_encoding_map)



with open('Jamming_Type.json', 'w', encoding='utf-8') as f:
    json.dump(Jamming_Type, f, indent=4, ensure_ascii=False)

with open('Jammer_carrierFrequency.json', 'w', encoding='utf-8') as f:
    json.dump(Jammer_carrierFrequency, f, indent=4, ensure_ascii=False)

with open('Jamming_Params.json', 'w', encoding='utf-8') as f:
    json.dump(Jamming_Params, f, indent=4, ensure_ascii=False)

with open('Radar_carrierFrequency.json', 'w', encoding='utf-8') as f:
    json.dump(Radar_carrierFrequency, f, indent=4, ensure_ascii=False)

with open('Radar_Params.json', 'w', encoding='utf-8') as f:
    json.dump(Radar_Params, f, indent=4, ensure_ascii=False)

with open('SINR.json', 'w', encoding='utf-8') as f:
    json.dump(SINR, f, indent=4, ensure_ascii=False)




import json

# 读取原始 JSON 文件
with open("Jamming_Type.json", "r", encoding="utf-8") as f:
    jamming_data = json.load(f)

# 提取 Jamming_Type 字段并进行编码
unique_types = {}
encoded_types = []

counter = 0
for item in jamming_data:
    jt = item["Jamming_Type"]
    if jt not in unique_types:
        unique_types[jt] = counter
        counter += 1
    encoded_types.append(unique_types[jt])

# 保存编码后的结果为新 JSON 文件
with open("Encoded_Jamming_Type.json", "w", encoding="utf-8") as f:
    json.dump(encoded_types, f, ensure_ascii=False, indent=4)











# s = df.iloc[45,2]
# params = get_Jamming_Params(s)
# print(params)


# s = df.iloc[0,3]
# params = get_Radar_carrierFrequency(s)
# print(params)


# s = df.iloc[0,4]
# params = get_Radar_Params(s)
# print(params)


# s = df.iloc[0,5]
# params = get_SINR(s)
# print(params)

#print(df.head())
# print("第一列数据：")
# print(df.iloc[:, 0])
# Interference = []
# for i in range(0, len(df)):
#     if(df.iloc[i,0] not in Interference):
#         Interference.append(df.iloc[i,0])
# print(Interference)