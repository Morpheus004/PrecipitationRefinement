import pandas as pd

# models_df = pd.DataFrame.from_dict({
#     "JJASNoScalingh32_64_32" : {
#         "kernel_size" : 5,
#         "hidden_channels" : [32,64,32],
#         "year_trained" : '2018',
#         "best_checkpoint_path" : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/260815684233579231/535735d042694da2b1da4c60f0a13f42/artifacts/best_model_checkpoint/best_model_epoch_19.pth1nmw8wtv.pth'
#     },
#     "JJASMaxScalingh32_64_32" : {
#         "kernel_size" : 3,
#         "hidden_channels" : [32,64,32],
#         "year_trained" : '2018',
#         "best_checkpoint_path" : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/196011945286209413/1178381b05ad4251bed6a330e45be89f/artifacts/best_model_checkpoint/best_model_epoch_19.pth19meyytt.pth'
#     },
#     "JJASNoScalingh64_128_64" : {
#         "kernel_size" : 5,
#         "hidden_channels" : [64,128,64],
#         "year_trained" : '2018',
#         "best_checkpoint_path" : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/861560316923204160/bca60636557547fca1cbab312cf0290f/artifacts/best_model_checkpoint/best_model_epoch_21.pthdtjkfu6n.pth'
#     },
#     "JJASMaxScalingh64_128_64" : {
#         "kernel_size" : 5,
#         "hidden_channels" : [64,128,64],
#         "year_trained" : '2018',
#         "best_checkpoint_path" : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/325894012973976081/cff81d37a3f7496892aced162f7fcf51/artifacts/best_model_checkpoint/best_model_epoch_21.pthjr6w9f3y.pth'
#     },
#     "JJASNoScalingh128_256_128" : {
#         "kernel_size" : 5,
#         "hidden_channels" : [128,256,128],
#         "year_trained" : '2018',
#         "best_checkpoint_path" : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/321967748427509393/ac9942442472460cb9c017ab3b33bb5d/artifacts/best_model_checkpoint/best_model_epoch_21.pthl9ulcg8h.pth'
#     },
#     "JJASMaxScalingh128_256_128" : {
#         "kernel_size" : 5,
#         "hidden_channels" : [128,256,128],
#         "year_trained" : '2018',
#         "best_checkpoint_path" : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/236425716617773343/0adcebe189ee4a508d390cafb4881d88/artifacts/best_model_checkpoint/best_model_epoch_21.ptha3olj1no.pth'
#     },
#     "UNet_MaxScale" : {
#         'year_trained' : '2018',
#         'epoch' : 21,
#         'batch_size' : 64,
#         'best_checkpoint_path' : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/441162194934542005/60e9c1cf7efe442d821707f8839f90f5/artifacts/best_model_checkpoint/best_model_epoch_21.pthx2u2is44.pth',
#     },
#     "UNet_NoScale" : {
#         'year_trained' : '2018',
#         'epoch' : 21,
#         'batch_size' : 64,
#         'best_checkpoint_path' : '/scratch/IITB/monsoon_lab/24d1236/pratham/Model/mlruns/793580818932410493/51c3634ba529441798da964c5a66d290/artifacts/best_model_checkpoint/best_model_epoch_21.pthgg39l_lh.pth',
#     }
# },orient='index')
models_df = pd.read_csv('/scratch/IITB/monsoon_lab/24d1236/pratham/Model/models.csv')
