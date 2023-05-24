from sklearn.model_selection import ParameterSampler
import pickle
import json

param_grid = {'neurons':[10,20,30,40,50,60,70,80],
              'n_hidden': [2,3,4,5,6,7,8,9,10],
              'learning_rate':[0.001,0.0001,0.00001],
              'learning_rate_decay':[1, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8],
              'batch_size':[64,128,256,512,1024]}
param_list = list(ParameterSampler(param_grid, n_iter=1000,
                                   random_state=7))

with open ('param_list.pkl', 'wb') as pick:
    pickle.dump(param_list, pick)

json_object = json.dumps(param_list, indent=4)
 
# Writing to sample.json
with open("param_list.json", "w") as outfile:
    outfile.write(json_object)


