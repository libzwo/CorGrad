import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

# PACS 
file1_path = "/export/home/zyj/swad/train_output/PACS/231022_15-19-59_PACS/change_rate_history.pkl"
file2_path = "/export/home/zyj/swad/train_output/PACS/231012_13-59-04_PACS/change_rate_history.pkl"
# 0.46

# VLCS
# file1_path = "/export/home/zyj/swad/train_output/VLCS/231022_15-20-13_VLCS/change_rate_history.pkl"
# file2_path = "/export/home/zyj/swad/train_output/VLCS/231012_13-58-52_VLCS/change_rate_history.pkl"
# 0.57

# OfficeHome
# file1_path = "/export/home/zyj/swad/train_output/OfficeHome/231022_15-16-11_OfficeHome/change_rate_history.pkl"
# file2_path = "/export/home/zyj/swad/train_output/OfficeHome/231012_13-59-14_OfficeHome/change_rate_history.pkl"
# 0.41

# # TerraInc
# file1_path = "/export/home/zyj/swad/train_output/TerraIncognita/231022_15-16-19_TerraIncognita/change_rate_history.pkl"
# file2_path = "/export/home1/zyj/swad/train_output/TerraIncognita/231012_13-59-28_TerraIncognita/change_rate_history.pkl"
# 0.41
# DomainNet
# file1_path = "/export/home/zyj/swad/train_output/DomainNet/231013_02-04-54_DomainNet/change_rate_history.pkl"
# file2_path = "/export/home/zyj/swad/train_output/DomainNet/231012_14-00-04_DomainNet/change_rate_history.pkl"
# 0.43


# 定义柱状图的宽度
bar_width = 0.25

with open(file1_path, 'rb') as f1:
    loaded1_list = pickle.load(f1)
# categories = ['0-500', '501-1000', '1001-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500','3501-4000','4001-4500','4501-5000']
categories = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35','35-40','40-45','45-50']
# categories = ['0-1500', '1501-3000', '3001-4500', '4501-6000', '6001-7500', '7501-9000', '9001-10500','10501-12000','12001-13500','13501-15000']
chunks_1 = [loaded1_list[i:i+500] for i in range(0, len(loaded1_list)-1, 500)]
values_1 = [sum(chunk) / len(chunk) for chunk in chunks_1]

# 生成 x 坐标
x = np.arange(len(categories))

with open(file2_path, 'rb') as f2:
    loaded2_list = pickle.load(f2)
# loaded2_list = [torch.tensor(tensor) for tensor in loaded2_list]
loaded2_list = [tensor.to("cpu") for tensor in loaded2_list]
chunks_2 = [loaded2_list[i:i+500] for i in range(0, len(loaded2_list)-1, 500)]
values_2 = [sum(chunk) / len(chunk) for chunk in chunks_2]

x_labels = [i * 500 for i in range(len(values_1))]

plt.bar(x - bar_width/2, values_1, bar_width, label='ours', color='b')
plt.bar(x + bar_width/2, values_2, bar_width, label='agr-sum', color='r')


plt.xlabel('Iterations',fontsize=24)
plt.ylabel('Average Modification Frequency',fontsize=24)
# 设置 x 轴刻度标签
plt.xticks(x, categories, rotation=30, fontsize=16)
# plt.xticks(x, categories,fontsize=12)
plt.legend(loc=1, fontsize=16)
# plt.grid(True)
plt.ylim((0, 0.46))
y_min, y_max = plt.ylim()
x_min, x_max = plt.xlim()
plt.text(x_max, -0.1 * y_max, 'X300', fontsize = 16)
plt.tight_layout()
plt.savefig(file1_path + '_frequency.pdf')  # 指定文件名和格式
