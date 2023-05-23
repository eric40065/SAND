# 1. Generate the data
## 1.1 Simulation data
The data generating code is in `R_code/SimulationAnalysis.R`. Lines 3 to 470 defines functions that generate simulated data and perform the analysis using PACE, 1DS, and MICE. Use line 473 to create the simulated dataset.

## 1.2 Real data
The UK electricity dataset can be downloaded at https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london. Please download the data in `halfhourly_dataset` folder and put the downloads in `Data/UK_Raw/halfhourly_dataset`. Our code will read the data in this path.

The real data is preprocessed by functions from `R_code/UKDataAnalysis.R`. Lines 3 to 272 defines functions that preprocess the data and perform the analysis using PACE, 1DS, and MICE. Use lines 275 to 285 to create the dataset.

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
