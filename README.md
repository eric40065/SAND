This README provides an overview of the project structure and instructions for generating and analyzing data, training networks, and evaluating on the testing dataset.

# 1. Generate the Data
In this project, we use the R programming language for data preprocessing and simulation. Python is employed solely for training the machine learning models.

## 1.1 Simulation Data
To generate simulation data, follow these steps:

1. Locate the data generating code in `R_code/SimulationAnalysis.R`.
2. Run lines 3 to 470 in the code to define the functions that generate simulated data and perform the analysis using PACE, 1DS, and MICE.
3. Utilize line 473 to create the simulated dataset.

## 1.2 Real Data
To work with real data, please obtain the UK electricity dataset from [this link](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london). Then, follow these steps:

1. Download the data from the `halfhourly_dataset` folder from the above link.
2. Place the downloaded files in `Data/UK_Raw/halfhourly_dataset`. Our code will read the data from this path.
3. Preprocess the real data using the functions provided in `R_code/UKDataAnalysis.R`.
4. Run lines 3 to 272 in the code to define the functions for data preprocessing and analysis using PACE, 1DS, and MICE.
5. Create the dataset by executing lines 275 to 285.

# 2. Train Networks
This section covers the training of two types of networks: transformer-based and conditional neural process.

## 2.1 Transformer-based Network
To train a transformer-based method (vanilla transformer, transformer with penalties, SNAD), perform the following steps:

1. Open `py_code/train.py`.
2. Specify the data to be analyzed in lines 13, 17, and 18.
3. Define the device in line 21.
4. Configure the model in lines 26, 30, and 41.
5. Once the above arguments are defined, use `py_code/train.py` to train the network.
6. The best checkpoint will be stored in the `Checkpoints` folder.

## 2.2 Conditional Neural Process
For training a conditional neural process, follow these instructions:

1. Launch `py_code/NPtrain.py`.
2. Specify the data to be analyzed in lines 14, 18, and 19.
3. Define the device in line 22.
4. Set up the model in lines 26 to 35.
5. After defining the necessary arguments, use `py_code/NPtrain.py` to train the network.
6. The best checkpoint will be saved in the `Checkpoints` folder under the method name CNP.

# 3. Evaluate on the Testing Dataset
In this section, we evaluate the trained networks on the testing dataset.

# 3.1 Transformer-based Network
To obtain the mean squared error (MSE) and imputations for the training and testing datasets, follow these steps:

1. `Execute py_code/ImputeAllData.py`.
2. Use the `evaluation()` function to retrieve the MSE of the training and testing datasets and store the imputations in the `ImputedData` folder.
3. The function call should be in the following format: `result = evaluation(data_name, iidt, output_structure="SAND", d=60, split=(90, 5, 5), cuda_device="cpu")`.
4. Adjust the arguments `data_name` and `iidt` to match your data.
5. Specify the desired model structure using the `output_structure` argument (options: `SAND`, `Vanilla`, `SelfAtt`).
6. Set the embedding dimensionality `d` to match the value used in `py_code/train.py`.
7. Define the proportion of training/validation/testing datasets in the `split` argument, consistent with the settings in `py_code/train.py`.

## 3.2 Conditional Neural Process

# 4. Get estimators from PACE/1DS/MICE.
In both `R_code/SimulationAnalysis.R` and `R_code/UKDataAnalysis.R`, we define the function `do_analysis()` to run the analyses.
- Usage:
  * `result = do_analysis()`
- Inputs:
  * Arguments `data_name_list` and `dense_sparse_list` specifies the dataset:
  	* `data_name_list` from `R_code/SimulationAnalysis.R` takes a vector of data names. The options are `HighDim_E`, `LowDim_G`, `HighDim_G`, `LowDim_E`, `LowDim_T`, `HighDim_T`.
   	* `data_name_list` from `R_code/UKDataAnalysis.R` takes `UK` as the input.
   	* `dense_sparse_list` from `R_code/SimulationAnalysis.R` is by default `list(c("dense", "w_error"), c("sparse", "w_error"), c("dense", "wo_error"), c("sparse", "wo_error"))`.
   	* `dense_sparse_list` from `R_code/UKAnalysis.R` is by default `c("dense", "sparse")`.
  * Argument `iidt_list` from both functions is by default `c("IID", "NonIID")`, indicating that both iid and non-iid cases are analyzed.
  * Arguments `do_PACE`, `do_1DS`, and `do_MICE` from both functions are `TRUE` by default, indicating all three methods will be run.
  * Argument `split` specify the proportion of training/validation/testing data.
- Outputs:
  * `result` contains a list of `MSE` storing the MSEs of training and testing data under different settings.
 

  

Thank you for participating in our research project! If you have any questions, feel free to reach out to us.

