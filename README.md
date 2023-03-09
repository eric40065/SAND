# FunctionalTransformer

Zip of data, checkpoints, and plots are in: https://drive.google.com/drive/folders/1apdcbxM6sMPA1CRFUHDffTtlfgmGetRJ?usp=share_link

## Generate the data
The data generating code is in `R_code/Simulation`. Run `get_all_data()` from `get_all_data.R` to generate all simulation data.

## Train the model -- Use train.py
* If you want to train it on your local machine, set `my_computer = True` and change the `server_specified_folder` to your path. 
* If you want to train it on server, set `my_computer = False` and change the `server_specified_folder` to your path. 

Notice that `server_specified_folder` should contain a folder named `TransFD` which contains `py_code`.

Use 
* line 18 to change the data you would like to analysis. 
* line 20 to change the `output_structure`. Options are `Vanilla`, `SelfAtt`, and `DiffSelfAtt`.

To run it on server, use `submit Trans_run_HighDim_G_DiffSelfAtt_l2w_False_False.sh` on the `shell_files` folder and it shall follow the form: `submit Trans_run_DATA NAME_OUTPUT STRUCTURE_DENOISE METHOD_DENSE OR NOT_ERROR OR NOT.sh`. All `submit` command can be found in `run_all.txt`.

### Model configuration
Line 50 -- 53 defines the model.

## Some results
![Imputation of different methods](https://github.com/eric40065/FunctionalTransformer/blob/main/Rplot.png)
