from Versions import jjas_no_scaling
from logger_config import setup_logger
from cust_types import ScalingType

if __name__=='__main__':
    # hc=[64,128,64]
    # hc=[128,256,128]
    # bs=[64,128,256,512]
    # for i in bs:
    #     jjas_no_scaling.jjas_main(logger=logger,batch_size=i,hidden_channels=hc,scaling_type='GlobalMax',kernel_size=5)
    #     jjas_no_scaling.jjas_main(logger=logger,batch_size=i,hidden_channels=hc,scaling_type='GlobalMax',kernel_size=3)
    #     jjas_no_scaling.jjas_main(logger=logger,batch_size=i,hidden_channels=hc,scaling_type='NoScaling',kernel_size=5)
    #     jjas_no_scaling.jjas_main(logger=logger,batch_size=i,hidden_channels=hc,scaling_type='NoScaling',kernel_size=3)
    logger = setup_logger('gmtraj')
    scalingtype: ScalingType = 'GlobalMax'
    jjas_no_scaling.jjas_main(logger=logger,batch_size=64,kernel_size=3,scaling_type=scalingtype,hidden_channels=[32,64,32])

    logger = setup_logger('nstraj')
    scalingtype: ScalingType = 'NoScaling'
    jjas_no_scaling.jjas_main(logger=logger,batch_size=64,kernel_size=3,scaling_type=scalingtype,hidden_channels=[32,64,32])
    
    # K-Fold Cross Validation Usage
    # Uncomment the following line to run k-fold cross validation
    # 
    # # Run 5-fold cross validation with kernel size 5
    # fold_results, summary_stats = jjas_no_scaling.jjas_kfold(
    #     logger=logger, 
    #     k_folds=5, 
    #     kernel_size=5, 
    #     num_epochs=50
    # )
    # 
    # # Results are automatically logged to MLflow run "kfold conclusion"
    # print(f"Mean validation loss: {summary_stats['mean_val_loss']:.6f}")
    # print(f"Best validation loss: {summary_stats['best_val_loss']:.6f}")
