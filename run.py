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
    

    bs=[64,128]

    # for i in [128]:
    #     hc=[128,256]
    #     kernel_size=[3,5]
    #     for k in kernel_size:
    #         for L in range(10,15):
    #             model = TrajGRU(1,hc,k,L)
    #             scaling_type : ScalingType = 'NoScaling'
    #             run_name=f"1_Traj_run_b{i}_h{'_'.join(map(str,hc))}_L{L}_k{k}"
    #             experiment_name=f"Traj_JJAS_{scaling_type}_h{'_'.join(map(str,hc))}"
    #             description=f"First run with kernel size {k} and batch size {i} with hidden channels as {'_'.join(map(str,hc))} and L as {L}"
    #             logger = setup_logger(f'traj_noscale')
    #             log_file = "/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log"
    #             jjas.jjas_main(logger=logger,
    #                            model=model,
    #                            batch_size=i,
    #                            experiment_name=experiment_name,
    #                            run_name=run_name,
    #                            description=description,
    #                            hidden_channels=hc,
    #                            scaling_type=scaling_type,
    #                            kernel_size=k,
    #                            log_file=log_file
    #                            )
    #             del model
    #             gc.collect()
    #             torch.cuda.empty_cache()   
    #
    # for i in bs:
    #     hc=[128,256]
    #     kernel_size=[3,5]
    #     for k in kernel_size:
    #         for L in range(9,15):
    #             model = TrajGRU(1,hc,k,L)
    #             scaling_type : ScalingType = 'GlobalMax'
    #             run_name=f"1_Traj_run_b{i}_h{'_'.join(map(str,hc))}_L{L}_k{k}"
    #             experiment_name=f"Traj_JJAS_{scaling_type}_h{'_'.join(map(str,hc))}"
    #             description=f"First run with kernel size {k} and batch size {i} with hidden channels as {'_'.join(map(str,hc))} and L as {L}"
    #             logger = setup_logger(f'traj_scale')
    #             log_file = "/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log"
    #             jjas.jjas_main(logger=logger,
    #                            model=model,
    #                            batch_size=i,
    #                            experiment_name=experiment_name,
    #                            run_name=run_name,
    #                            description=description,
    #                            hidden_channels=hc,
    #                            scaling_type=scaling_type,
    #                            kernel_size=k,
    #                            log_file=log_file
    #                            )
    #             del model
    #             gc.collect()
    #             torch.cuda.empty_cache()   


    for i in bs:
        hc=[256,512]
        kernel_size=[3,5]
        for k in kernel_size:
            for L in range(9,15):
                model = TrajGRU(1,hc,k,L)
                scaling_type : ScalingType = 'NoScaling'
                run_name=f"1_Traj_run_b{i}_h{'_'.join(map(str,hc))}_L{L}_k{k}"
                experiment_name=f"Traj_JJAS_{scaling_type}_h{'_'.join(map(str,hc))}"
                description=f"First run with kernel size {k} and batch size {i} with hidden channels as {'_'.join(map(str,hc))} and L as {L}"
                logger = setup_logger(f'traj_noscale')
                log_file = "/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log"
                jjas.jjas_main(logger=logger,
                               model=model,
                               batch_size=i,
                               experiment_name=experiment_name,
                               run_name=run_name,
                               description=description,
                               hidden_channels=hc,
                               scaling_type=scaling_type,
                               kernel_size=k,
                               log_file=log_file
                               )
                del model
                gc.collect()
                torch.cuda.empty_cache()   

    for i in bs:
        hc=[256,512]
        kernel_size=[3,5]
        for k in kernel_size:
            for L in range(9,15):
                model = TrajGRU(1,hc,k,L)
                scaling_type : ScalingType = 'GlobalMax'
                run_name=f"1_Traj_run_b{i}_h{'_'.join(map(str,hc))}_L{L}_k{k}"
                experiment_name=f"Traj_JJAS_{scaling_type}_h{'_'.join(map(str,hc))}"
                description=f"First run with kernel size {k} and batch size {i} with hidden channels as {'_'.join(map(str,hc))} and L as {L}"
                logger = setup_logger(f'traj_scale')
                log_file = "/scratch/IITB/monsoon_lab/24d1236/pratham/Model/model_training.log"
                jjas.jjas_main(logger=logger,
                               model=model,
                               batch_size=i,
                               experiment_name=experiment_name,
                               run_name=run_name,
                               description=description,
                               hidden_channels=hc,
                               scaling_type=scaling_type,
                               kernel_size=k,
                               log_file=log_file
                               )
                del model
                gc.collect()
                torch.cuda.empty_cache()   



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
