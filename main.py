import os
from torch.utils.data import Dataset,DataLoader
from glob import glob
import xarray as xr
import pandas as pd
from DatasetVariations import T4DatasetNoScaling,T4DatasetMinMaxScalerOverall,T4DatasetNoScaling
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from model import RefinementModel
from torchinfo import summary
from train_refinement_model import train_refinement_model
from file_mappings import T_4_file_to_file_mapping_training,T_4_file_to_file_mapping_validation
import time 

start_time=time.time()
# training_data=T4DatasetNoScaling(T_4_file_to_file_mapping_training,data='forecast')
# testing_data=T4DatasetNoScaling(T_4_file_to_file_mapping_validation,data='forecast')
training_data=T4DatasetMinMaxScalerOverall(T_4_file_to_file_mapping_training,data='forecast',initialize_global_min_max=True)
testing_data=T4DatasetMinMaxScalerOverall(T_4_file_to_file_mapping_validation,data='forecast',initialize_global_min_max=False,global_max=training_data.max)

train_loader=DataLoader(training_data,64,shuffle=True)
val_loader=DataLoader(testing_data,64,shuffle=True)

model=RefinementModel()

summary(model,(1,2,1,36,41))

model,optimizer,history,num_epochs=train_refinement_model(model,train_loader,val_loader,num_epochs=500,checkpoint_save_dir='/scratch/IITB/monsoon_lab/24d1236/pratham/Model/TrainedModels/third_run_global_min_max/')

end_time=time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

checkpoint = { 
    'epoch': num_epochs,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'history': history 
    # 'lr_sched': lr_sched
}
torch.save(checkpoint, '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/TrainedModels/third_run_global_min_max/checkpoint501.pth')
print("Checkpoint saved")
# First Run
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset,DataLoader
# from file_mappings import T_4_file_to_file_mapping,T_4_imergfile_to_erafile_mapping
# from DatasetVariations import T4DatasetNoScaling,T4DatasetMinMaxScalerOverall,T4DatasetNoScaling
# from train_refinement_model import train_refinement_model
# from model import RefinementModel
# import pickle

# training_dataset=T4DatasetNoScaling(T_4_file_to_file_mapping,data='forecast')
# testing_dataset=T4DatasetNoScaling(T_4_imergfile_to_erafile_mapping,data='era5')

# train_loader=DataLoader(training_dataset,16,shuffle=True)
# test_loader=DataLoader(testing_dataset,16,shuffle=True)

# model=RefinementModel()

# trained_model,history = train_refinement_model(model,train_loader,test_loader,num_epochs=50)

# torch.save(trained_model.state_dict(),'precip_refinement_model1.pth')

# with open('history.pkl', 'wb') as f:
#     pickle.dump(history, f)
