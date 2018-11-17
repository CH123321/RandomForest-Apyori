# coding: utf-8

# ### 导入库以及Test2数据
import pandas as pd  
import numpy as np
from apyori import apriori  

dataset = pd.read_csv("./src/Test2_data.csv")
items = dataset.columns
length = len(dataset)

D=[]
for i in range(length):
    # 只保留value为 1 或 True 的 item
    tempTF = dataset.loc[i]==1
    temp_transa = items[tempTF].tolist()
    D.append(temp_transa)

f = open("./output_q2/dataList.txt","w")
for i in range(length):
    f.write("%s\n" %D[i])
f.close()


# ### 使用apriori function获取 Rule

# 获取 support>=0.1, confidence >= 0.7 的Rule, 最大长度为3
association_rules = apriori(D, min_support=0.1, min_confidence=0.7,max_length =3) 
# 将iterator 转化为 list
association_results=list(association_rules)

# 筛选出 等式右侧为 "Label" 的 Rule
ruleLen = len(association_results)
# 将 Rule 中的 元素保存到 list中
resultList=[]
for i in range(ruleLen):
    if association_results[i].ordered_statistics[0].items_add == frozenset(['Label']):
        temp = list (association_results[i].ordered_statistics[0].items_base)
        temp.append("Label")
        resultList.append( temp )


# ## 输出 结果到 output_q2 文件夹下 的 output.txt
f = open("./output_q2/output.txt","w")
flag=0
for rule in resultList:
    if not flag:
        if len(rule)>2:
            flag=1
    
    if flag:
        f.write("Rule:\t{}, {} -> {}.\n".format(rule[0],rule[1], rule[2]))
    else:
        f.write("Rule:\t{} -> {}.\n".format(rule[0],rule[1]))
f.close()

