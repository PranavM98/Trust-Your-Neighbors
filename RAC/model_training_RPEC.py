import pandas as pd

import models.RPEC as rac_joint
import os
import re
import pandas as pd
import os
import utils


import pandas as pd

# import models.mlp as mlp
import os
import re
import pandas as pd
import os
import argparse
import utils





parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, default='1', help='1,2, 3')
args = parser.parse_args()

folder="add folder where models exist here"

results = []

for data_text_split in ['institutional','temporal']:#'original'

    for model in ['mmanet']:
        for image_model in ['ctnet']:
            for input in ['report']:
                for t in ['1','2','3','4','5','6','7','8']:
                    trial=t

                    file_path=folder+model+'/'+image_model+"_supcon_original_"+f"{input}_"+f"{trial}"
                    try:
                        final_folder=utils.get_last_epoch_data(file_path)
                        test_auroc, auroc_class_0, auroc_class_1, prob,f1, sensitivity, results_df=rac_joint.rac_joint_main(final_folder, k=7)

                        result_dict = {
                            'datasplit': data_text_split,
                            'model': model,
                            'image_model': image_model,
                            'loss': 'supcon',
                            'input': input,
                            'trial': trial,
                            'test_auroc': test_auroc,
                            'class_0_auroc':auroc_class_0,
                            'class_1_auroc':auroc_class_1,
                            'probabilities':prob,
                            'f1':f1,
                            'Sensitivity':sensitivity

                        }
                        print(result_dict)
                        results.append(result_dict)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print("An error occurred:", e)  
                        print("Not in trial")

results_df = pd.DataFrame(results)
results_df.to_csv(f"results/{model}_rac_joint_results_{data_text_split}.csv", index=False)
