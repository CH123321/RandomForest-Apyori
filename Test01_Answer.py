# coding: utf-8
# ### 加载库
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ### 导入Test1数据
full_features = pd.read_table("./src/Test1_features.dat",sep = ',',header=None)
labels =  pd.read_table("./src/Test1_labels.dat",sep = ',', header=None)


# ## Random Forest 模型
# >由于数据集中 features 数目较多, 选用Random Forest可以有效地降低 irrelevant features 的影响
# ## 使用Grid Search来优化参数
# 选取的两个待优化参数为:  
# * n_estimators: 树的数量 50~100
# * max_depth: 树的最大深度 5~50
# 这两个参数有低变高时都可以有效地 降低 training loss, 但如果太高了, 就会导致 overfitting, 所有需要找到一个合适的取值.
# 
# 本次Grid Search 采用了 5-fold cross validation来优化参数.  
# 
# 注意: 本次 Training过程使用了 8 个 cpu core, 总耗时约 9 分钟.  
#     电脑CPU较差请勿运行, 请直接使用 pickle load 已保存在 **output_q1** 文件夹下的 *rf_GridSearchCV.pickle* 文件

tuneParams ={"n_estimators":range(50,151,10), "max_depth":range(5,51,5) }
rf_GridSearch= GridSearchCV(
    estimator =RandomForestClassifier(min_samples_split=100, min_samples_leaf=20, max_features='sqrt', random_state=1117),
    param_grid =tuneParams, scoring='roc_auc', cv=5, n_jobs=-1, return_train_score=True)

rf_GS = rf_GridSearch.fit(full_features.values,labels.values.ravel())

# with open('./output_q1/rf_GridSearchCV.pickle', 'rb') as f:
#     rf_GS = pickle.load(f)
#     print (rf_GS)

f = open("./output_q1/output.txt","w")
f.write("By Grid Search, we got the best parameters {}.\nThe model with best parameters has the highest CV AUC value: {}.\n".format(
    rf_GS.best_params_,rf_GS.best_score_))
best_rf =rf_GS.best_estimator_
f.write ("Below is the best Random Forest model's information:\n{}\n".format(best_rf))

f.close()


# ### 使用pickle 保存已训练好的模型

with open('./output_q1/rf_GridSearchCV.pickle', 'wb') as f2:
    pickle.dump(rf_GS, f2)

with open('./output_q1/best_RandomForest_model.pickle', 'wb') as f1:
    pickle.dump(best_rf, f1)

	
# ## 提取 参数与AUC 信息并画图
length = len(rf_GS.cv_results_['params'])
x = [rf_GS.cv_results_['params'][i]["n_estimators"] for i in range(length)]
y = [rf_GS.cv_results_['params'][i]["max_depth"] for i in range(length)]
z = rf_GS.cv_results_['mean_test_score']

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x,y,z,cmap="Blues",linewidth=0.2, antialiased=True)
#ax.set_zlabel("AUC Value")
ax.text(155, 50,0.99255,"AUC Value")
plt.xlabel('Number of Trees')
plt.ylabel('Max Depth')

fig.savefig("./output_q1/result.png")