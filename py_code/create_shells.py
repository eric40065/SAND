##### Transformer
### Vanilla
# Simulation
for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import os
os.chdir("/Users/eric/Desktop/UCD/TransFD/shell_files")
data_name_list = list(("LowDim_G", "HighDim_G", "LowDim_E", "HighDim_E", "LowDim_T", "HighDim_T", "UK", "Framingham"))
special_data = "HighDim_E"

for data_name in data_name_list:
    output_list = list(("Vanilla", "SelfAtt", "DiffSelfAtt")) if data_name == special_data else ["DiffSelfAtt"]
    error_dense_list = list((("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")))
    error_dense_list = [("True", "True")] if data_name == "Framingham" else error_dense_list
    error_dense_list = list((("True", "True"), ("True", "False"))) if data_name == "UK" else error_dense_list
    for output in output_list:
        denoise_method_list = list(("None", "l2w", "l1w", "l2o", "TVo")) if output == "Vanilla" and data_name == special_data else ["l2w"]
        for denoise_method in denoise_method_list:
            for error, data_is_dense in error_dense_list:
                # aa
                filename = "./Trans_run_" + data_name + "_" + output + "_" + denoise_method + "_" + data_is_dense + "_" + error + ".sh"
                lines = "python3 ../py_code/train.py " + data_name + " " + output + " " + denoise_method + " " + data_is_dense + " " + error
                with open (filename, 'w') as rsh:
                    rsh.write("#! /bin/bash\n" + lines)
                
## create multiple shell files
os.chdir("/Users/eric/Desktop/UCD/TransFD/shell_files")
data_name_list = list(("LowDim_G", "HighDim_G", "LowDim_E", "HighDim_E", "LowDim_T", "HighDim_T"))
for data_name in data_name_list:
    for data_is_dense in list((True, False)):
        for error in list((True, False)):
            filename = "CNP_run_" + data_name + "_" + str(data_is_dense) + "_" + str(error) + ".sh"
            lines = "python3 ../py_code/NPtrain.py " + data_name + " " + str(data_is_dense) + " " + str(error)
            with open (filename, 'w') as rsh:
                rsh.write("#! /bin/bash\n" + lines)
            
os.chdir("/Users/eric/Desktop/UCD/TransFD/shell_files")
for data_is_dense in list((True, False)):
    filename = "CNP_run_UK_" + str(data_is_dense) + ".sh"
    lines = "python3 ../py_code/NPtrain.py " + "UK" + " " + str(data_is_dense)
    with open (filename, 'w') as rsh:
        rsh.write("#! /bin/bash\n" + lines)
        
os.chdir("/Users/eric/Desktop/UCD/TransFD/shell_files")
filename = "CNP_run_Framingham_True.sh"
lines = "python3 ../py_code/NPtrain.py Framingham True"
with open (filename, 'w') as rsh:
    rsh.write("#! /bin/bash\n" + lines)
