# 1. Generate the data
## 1.1 Simulation data
The data generating code is in `R_code/SimulationAnalysis.R`. 
* Run lines 3 to 470 to define functions that generate simulated data and perform the analysis using PACE, 1DS, and MICE. 
* Use line 473 to create the simulated dataset.

## 1.2 Real data
The UK electricity dataset can be downloaded at https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london. Please download the data in `halfhourly_dataset` folder and put the downloads in `Data/UK_Raw/halfhourly_dataset`. Our code will read the data in this path.

The real data is preprocessed by functions from `R_code/UKDataAnalysis.R`. 
* Run lines 3 to 272 to define functions that preprocess the data and perform the analysis using PACE, 1DS, and MICE. 
* Use lines 275 to 285 to create the dataset.

# 2. Train networks
## 2.1 Transformer-based network
To train a transformer-based method (vanilla transformer, transformer with penalties, SNAD), use `py_code/train.py`.
* Lines 13, 17, and 18 specify the data to be analyzed.
* Line 21 defines the device.
* Lines 26, 30, and 41 defines the model.
After defining the above argument, use `py_code/train.py` to train the network. The best checkpoint will be stored at the `Checkpoints` folder.

## 2.2 Conditional Neural Process

# 3. Get estimators from PACE/1DS/MICE.
In both `R_code/SimulationAnalysis.R` and `R_code/UKDataAnalysis.R`, we define the function `do_analysis()` to run the analyses.
- Inputs:
  * Arguments `data_name_list` and `dense_sparse_list` specifies the dataset:
    ** `data_name_list` from `R_code/SimulationAnalysis.R` takes a vector of data names. The options are `HighDim_E`, `LowDim_G`, `HighDim_G`, `LowDim_E`, `LowDim_T`, `HighDim_T`.
    ** `data_name_list` from `R_code/UKDataAnalysis.R` takes `UK` as the input.
    ** `dense_sparse_list` from `R_code/SimulationAnalysis.R` is by default `list(c("dense", "w_error"), c("sparse", "w_error"), c("dense", "wo_error"), c("sparse", "wo_error"))`.
    ** `dense_sparse_list` from `R_code/UKAnalysis.R` is by default `c("dense", "sparse")`.
  * Argument `iidt_list` from both functions is by default `c("IID", "NonIID")`, indicating that both iid and non-iid cases are analyzed.
  * Arguments `do_PACE`, `do_1DS`, and `do_MICE` from both functions are `TRUE` by default, indicating all three methods will be run.
  * Argument `split` specify the proportion of training/validation/testing data.

  

![Imputation of different methods](https://github.com/eric40065/FunctionalTransformer/blob/main/Rplot.png)


