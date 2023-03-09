import numpy as np
from project import inference
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from config import *
config, out_dir, _ = load_config()
rng = np.random.RandomState(16)

def norm(input):
    sc = StandardScaler()
    sc.fit(input)
    out = sc.transform(input)
    return out, sc.scale_, sc.mean_

X = np.load(out_dir / "latent_features.npy")
Y = np.load(config["data_params"]["data_path"] + config["data_params"]["data_name"] + "_Y_all.npy")
Y_norm, Y_scale, Y_mean = norm(Y)
Y_date =Y_norm[:,0]
Y_location = Y_norm[:,2:]
idx = np.arange(Y.shape[0])
_,_,idx_train,idx_test = train_test_split(idx,idx,test_size=0.15,random_state=rng)



def eval_time_regression():
    model = RandomForestRegressor(random_state=rng)
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=rng)
    cv_results = cross_validate(model,X,Y_date,cv=cv,scoring=('r2','neg_mean_absolute_error'))
    mae = cv_results["test_neg_mean_absolute_error"].mean() * (-Y_scale[0])
    r2 = cv_results["test_r2"].mean()
    print(f"Date regression: Mean Absolute Error|{mae}, r2 score|{r2}")
    return

def eval_location_regression():
    model = RandomForestRegressor(random_state=rng)
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=rng)
    cv_results = cross_validate(model,X,Y_location,cv=cv,scoring=('r2','neg_mean_absolute_error'))
    mae = cv_results["test_neg_mean_absolute_error"].mean()
    r2 = cv_results["test_r2"].mean()*(-1)
    print(f"Location regression: Mean Absolute Error|{mae}, r2 score|{r2}")
    return

if __name__=="__main__":
    eval_time_regression()
    eval_location_regression()