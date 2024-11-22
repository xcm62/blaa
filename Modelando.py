from fastai.learner import *
from fastai.tabular.all import *
from fastai.vision.all import *
import tensorflow.keras
import matplotlib as plt
import sklearn
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import fastai.metrics
import torch

train_df = train_clean
test_df = test_clean

print(train_df.isnull().sum()) 
print(test_df.isnull().sum())

cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

dep_var = 'Sales'

df = train_df[cat_vars + cont_vars + [dep_var,'Date']].copy()

cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()

valid_idx = range(cut)

df[dep_var].head()

max_log_y = np.log(np.max(train_df['Sales']) * 1.2)
y_range = tensor([0, max_log_y]).to(default_device())

def rmspe(preds, targs):
    mask = targs != 0
    return (torch.sqrt(((preds[mask] - targs[mask]) / targs[mask])**2).mean()).item()
    
test_df = test_df.copy()

splits = IndexSplitter(valid_idx)(range_of(df))
dls = TabularDataLoaders.from_df(
    df, 
    #path=path, 
    procs=procs, 
    cat_names=cat_vars, 
    cont_names=cont_vars, 
    y_names=dep_var, 
    y_block=RegressionBlock(), 
    splits=splits
)
config = tabular_config(ps=[0.001, 0.01]) 

learn = tabular_learner(
    dls, 
    layers=[1000, 500], 
    config=config,  # Aquí se pasa la configuración personalizada
    y_range=y_range, 
    metrics= rmspe
)

learn.model

cont_names = dls.cont_names
print(f"Número de variables continuas: {len(cont_names)}")

lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

suggested_lrs = lrs.valley

learn.fit_one_cycle(5, lr_max=suggested_lrs, wd=0.2)