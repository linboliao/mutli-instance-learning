import os

import pandas as pd


def remove_content(s):
    # 找到第一个"-"的位置
    first_dash_index = s.find('-')
    # 找到最后一个".kfb"的位置
    last_kfb_index = s.rfind('.h5')

    # 如果这两个位置都找到了，就进行切割
    if first_dash_index != -1 and last_kfb_index != -1:
        # 切割字符串，保留第一个"-"之前和最后一个".kfb"之后的内容
        return s[:first_dash_index] + s[last_kfb_index:]
    else:
        # 如果没有找到，返回原字符串
        return s


df = pd.DataFrame(columns=['slide_id', 'label'])
folder_path = '/data2/yhhu/LLB/Data/前列腺癌数据/test/features/256/uni_v1/h5_files/h5_files/'

filename = os.listdir(folder_path)
for file in filename:
    name = file.replace(" ", "").split('无癌')[0] + '.h5'
    name = remove_content(name)
    new_data = pd.DataFrame({'slide_id': [name], 'label': [0]})
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, name))
    df = pd.concat([df, new_data])
# df.to_csv('test.csv')
#
# folder_path = '/data2/yhhu/LLB/Data/前列腺癌数据/HE/youai/'
# filename = os.listdir(folder_path)
# for file in filename:
#     name = file.replace(" ", "").split('有癌')[0] + '.kfb'
#     name = remove_content(name)
#     new_data = pd.DataFrame({'slide_id': [name], 'label': [1]})
#     os.rename(os.path.join(folder_path, file), os.path.join(folder_path, name))
#     df = pd.concat([df, new_data])
# df.to_csv('test.csv')

# for i in range(5):
#     df = pd.read_csv('/data2/yhhu/LLB/Data/前列腺癌数据/HE/labels/label.csv')
#     df_half = df.sample(frac=0.7)
#     df_half.to_csv(f'./labels/label_for_train_{i}.csv', index=False)
#     df_remaining = df.drop(df_half.index)
#     df_quarter = df_remaining.sample(frac=0.3)
#     df_quarter.to_csv(f'./labels/label_for_valid_{i}.csv', index=False)
#     df_remaining = df_remaining.drop(df_quarter.index)
#     df_remaining.to_csv(f'./labels/label_for_test_{i}.csv', index=False)
