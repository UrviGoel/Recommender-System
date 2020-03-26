
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import auc_score


#fetch data and format it
data = fetch_movielens()
train,test = data['train'], data['test']

#print testing and training data in string representation
print(repr(data['train']))
print(repr(data['test']))


#create models
alpha = 1e-05
epochs = 60
num_components = 32


warp_model = LightFM(loss='warp', no_components=num_components, max_sampled=200 , user_alpha=alpha, item_alpha=alpha)
bpr_model = LightFM(loss='bpr', no_components=num_components, max_sampled=200 , user_alpha=alpha, item_alpha=alpha)
warpkos_model = LightFM(loss='warp-kos', no_components=num_components, max_sampled=200 , user_alpha=alpha, item_alpha=alpha)
logistic_model = LightFM(loss='logistic', no_components=num_components, max_sampled=200 , user_alpha=alpha, item_alpha=alpha)



warp_auc = []
bpr_auc = []
warpkos_auc = []
logistic_auc = []


for epoch in range(epochs):
	warp_model.fit_partial(train, epochs=1)
	warp_auc.append(auc_score(warp_model,test,train_interactions=train).mean())


for epoch in range(epochs):
	bpr_model.fit_partial(train, epochs=1)
	bpr_auc.append(auc_score(bpr_model,test,train_interactions=train).mean())


for epoch in range(epochs):
	warpkos_model.fit_partial(train, epochs=1)
	warpkos_auc.append(auc_score(warpkos_model,test,train_interactions=train).mean())


for epoch in range(epochs):
	logistic_model.fit_partial(train, epochs=1)
	logistic_auc.append(auc_score(logistic_model,test,train_interactions=train).mean())



x= np.arange(epochs)
plt.plot(x,np.array(warp_auc))
plt.plot(x,np.array(bpr_auc))
plt.plot(x,np.array(warpkos_auc))
plt.plot(x,np.array(logistic_auc))

plt.legend(['WARP AUC','BPR AUC','WARP-KOS AUC','LOGISTIC AUC'], loc='upper right')
plt.show()

