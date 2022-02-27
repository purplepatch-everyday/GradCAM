#%%
from cProfile import label
from cv2 import normalize, threshold
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve,auc,confusion_matrix, roc_auc_score 
import matplotlib.pyplot as plt
import json

# ##### FIRST DATA #####
df1 = pd.read_csv("DTD/resize_resnet_texture.csv", names= ["values"],sep=" ")
df1['classification'] = 0
df1=df1[['classification','values']]

# print("df1:",df1.shape)
# print(df1.head())

##### Imagenet #####
# with open('imagenet/imagenet_resize.json','r') as f:
#     json_data=json.load(f)

# imagenetmax =[]
# for i in range(len(json_data['max'])):
#     imagenetmax.append(json_data['max'][i])

# # df2 = np.random.choice(imagenetmax, 2000)
# df2 = pd.DataFrame({"values":imagenetmax})
# print(df2.shape)
df2 = pd.read_csv("imagenet/resize_resnet_imagenet.csv", names =["values"],sep=" ")
df2['classification']= 1
df2 =df2[['classification','values']]

# print("df2: ",df2.size)
# print(df2.head())

##### concat df1,df2 #####
df3 = pd.concat([df1,df2])
# print(df3.head())
# print(df3.tail())
print(df3.shape)

########################################################

#df1 = imagenet o 
#df2 = imagenet 
#df3 = imagenet o + imagenet

########################################################

##### Precision-Recall Curve #####

y_true = np.array(df3["classification"])
# print(y_true)
y_scores = np.array(df3["values"])
# print(y_scores)

precision, recall, thresholds = precision_recall_curve(y_true,y_scores)

# print(precision)
# print(recall)
# print (thresholds)
##################################

# plt.plot(recall[:-1],precision[:-1])]
plt.title("ImageNet - O (Resize n Crop) Entropy")
plt.plot(recall, precision)
plt.xlabel("recall")
plt.ylabel("Precision")
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

#### Calculate AUC #####
AUPR_score = auc(recall, precision)
print("AUPR",AUPR_score)

AUROC_score = roc_auc_score(y_true,y_scores)
print("AUROC",AUROC_score)


# %%
