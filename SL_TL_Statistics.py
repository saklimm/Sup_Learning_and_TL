import torch
import csv

import matplotlib.pyplot as plt

# available models:

Completed_Experiments = ['3_Poissonian_xDnCNN','3_Gaussian_DnCNN']

Missing_Experiments = ['3_Poissonian_3_Gaussian_xDnCNN', '3_Poissonian_3_Gaussian_DnCNN',
                       '3_Poissonian_DnCNN', '3_Poissonian_xDnCNN',
                       '3_Gaussian_xDnCNN']


# not completed yet:

#3_Poissonian_DnCNN: Step 60
#3_Poissonian_xDnCNN

#3_Gaussian_DnCNN
#3_Gaussian_xDnCNN

# 3_Poissonian_3_Gaussian_xDnCNN


def get_SL(experiment_name,Data_Path):
    
    Training_Path = Data_Path + 'Training_Results/'
    
    
    LOSS_PATH = Training_Path + 'Losses/Loss_Step_500'
    PSNR_PATH = Training_Path + 'PSNR/'
    SSIM_PATH = Training_Path + 'SSIM/'
    
    LOSS = torch.load(LOSS_PATH).numpy()
    
    PSNR_SL = torch.load(PSNR_PATH + 'PSNR_Step_500' )[2,9::10].numpy()
    PSNR_SL_INIT = torch.load(PSNR_PATH + 'Initial_PSNR' )
    
    SSIM_SL = torch.load(SSIM_PATH + 'SSIM_Step_500' )[2,9::10].numpy()
    SSIM_SL_INIT = torch.load(SSIM_PATH + 'Initial_SSIM' )
    
    
    psnr_SL_amax = PSNR_SL.argmax()
    
    ########################
    # TEST SET
    ########################
    
    PSNR_TEST_INIT = 27.6108
    SSIM_TEST_INIT = 0.6368
    
    PSNR_TEST = torch.load(Data_Path + 'SUPERVISED_LEARNING_TEST/PSNR_TEST'  )[2,:].numpy()
    SSIM_TEST = torch.load(Data_Path + 'SUPERVISED_LEARNING_TEST/SSIM_TEST'  )[2,:].numpy()
    
    psnr_TEST_amax = PSNR_TEST.argmax()
    
    
    row = [experiment_name, '',
           PSNR_SL_INIT, PSNR_SL[psnr_SL_amax],
           SSIM_SL_INIT, SSIM_SL[psnr_SL_amax], '',
           PSNR_TEST_INIT, PSNR_TEST[psnr_TEST_amax],
           SSIM_TEST_INIT, SSIM_TEST[psnr_TEST_amax]
           ]
    return row
    

def get_TL(experiment_name,Data_Path,FT_LR):
    
    TL_PATH = Data_Path + 'TRANSFER_LEARNING_FT_LR=' + FT_LR + '/'
    
    
    LOSS_PATH = TL_PATH + 'Losses/Loss_Step_500'
    PSNR_PATH = TL_PATH + 'PSNR/'
    SSIM_PATH = TL_PATH + 'SSIM/'
    
    LOSS = torch.load(LOSS_PATH).numpy()
    
    PSNR_FT = torch.load(PSNR_PATH + 'FT_PSNR_Step_500' )[2,9::10].numpy()
    PSNR_FT_INIT = 27.1839
    SSIM_FT = torch.load(SSIM_PATH + 'FT_SSIM_Step_500' )[2,9::10].numpy()
    SSIM_FT_INIT = 0.5772

    
    
    PSNR_TEST = torch.load(PSNR_PATH + 'VAL_PSNR_Step_500' )[2,9::10].numpy()
    PSNR_TEST_INIT =  27.6108
    SSIM_TEST = torch.load(SSIM_PATH + 'VAL_SSIM_Step_500' )[2,9::10].numpy()
    SSIM_TEST_INIT = 0.6368

    
    
    
    psnr_FT_amax = PSNR_FT.argmax()
    psnr_TEST_amax = PSNR_TEST.argmax()
    
    ########################
    # TEST SET
    ########################
       
    
    row = [experiment_name, '',
           PSNR_FT_INIT, PSNR_FT[psnr_FT_amax],
           SSIM_FT_INIT, SSIM_FT[psnr_FT_amax], '',
           PSNR_TEST_INIT, PSNR_TEST[psnr_TEST_amax],
           SSIM_TEST_INIT, SSIM_TEST[psnr_TEST_amax]
           ]
    return row




col_names_SL= ['Experiment Name','',
               'Initial PSNR (Train_Set)', 'Mean PSNR (Train_Set)',
               'Initial SSIM (Train_Set)', 'Mean SSIM (Train_Set)', '',
               'Initial PSNR (Test_Set)', 'Mean PSNR (Test_Set)',
               'Initial SSIM (Test_Set)', 'Mean SSIM (Test_Set)',
               
]


col_names_TL= ['Experiment Name','','Initial PSNR (FT Set)',
            'Mean PSNR (FT Set)','Initial SSIM (FT Set)',
            'Mean SSIM (FT Set)','','Initial PSNR (Test Set)',
            'Mean PSNR (Test Set)','Initial SSIM (Test Set)','Mean SSIM (Test Set)'
]






# =============================================================================
# Specify LR of Experiments here
# =============================================================================

SL_LR = '0.0001'
FT_LR = '0.0001'


# =============================================================================
# Specify Completed Experiments here
# =============================================================================


Completed_Experiments = ['3_Poissonian_3_Gaussian_DnCNN','3_Gaussian_DnCNN']

Missing_Experiments = ['3_Poissonian_3_Gaussian_xDnCNN', '3_Poissonian_3_Gaussian_DnCNN',
                       '3_Poissonian_DnCNN', '3_Poissonian_xDnCNN',
                       '3_Gaussian_xDnCNN']


with open('NI_SL_TL_Results_LR='+LR+'.csv', mode='w') as tab:
    table = csv.writer(tab, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    table.writerow(['Best Results:'])
    table.writerow([])
    
    table.writerow(['Supervised Learning:'])
    table.writerow([])
    
    table.writerow(col_names_SL)
    
    for experiment_name in Completed_Experiments:
        Data_Path= 'Sup_and_TL_Learning_LR='+SL_LR + '/' + experiment_name +'/'
        
        table.writerow( get_SL(experiment_name,Data_Path) )

    table.writerow([])
    table.writerow([])
    table.writerow(['Transfer Learning:'])
    table.writerow([])
    
    table.writerow(col_names_TL)  
    
    for experiment_name in Completed_Experiments:
        Data_Path= 'Sup_and_TL_Learning_LR='+FT_LR + '/' + experiment_name +'/'
        
        table.writerow( get_TL(experiment_name,Data_Path,FT_LR) )

    tab.close()
    









    