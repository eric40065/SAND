This README provides an overview of the project structure and instructions for generating and analyzing data, training networks, and evaluating on the testing dataset.

# 1. Generate the Data
In this project, we use the R programming language for data preprocessing and simulation. Python is employed solely for training and evaluating the machine learning models.

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
4. Configure the model in lines 26, 30, 41, and 46.
5. Once the above arguments are defined, run `py_code/train.py` to train the network.
6. The best checkpoint will be stored in the `Checkpoints` folder.

## 2.2 Conditional Neural Process
For training a conditional neural process, follow these instructions:

1. Launch `py_code/NPtrain.py`.
2. Specify the data to be analyzed in lines 14, 18, and 19.
3. Define the device in line 22.
4. Set up the model in lines 26 to 35.
5. After defining the necessary arguments, run `py_code/NPtrain.py` to train the network.
6. The best checkpoint will be saved in the `Checkpoints` folder under the method name CNP.

# 3. Evaluate on the Testing Dataset
In this section, we evaluate the trained networks on the testing dataset.

## 3.1 Transformer-based Network
To obtain the mean squared error (MSE) and imputations for the training and testing datasets using the transformer-based network, please follow these steps:

1. Execute the `py_code/ImputeAllData.py` script.
2. Use the `evaluation()` function to retrieve the MSE of the training and testing datasets and store the imputations in the `ImputedData` folder.
  * The function call should be in the following format: `result = evaluation(data_name, iidt, output_structure="SAND", d=60, split=(90, 5, 5), cuda_device="cpu")`.
  * Adjust the `data_name` and `iidt` arguments to match your specific data.
  * Specify the desired model structure using the `output_structure` argument (options: `SAND`, `Vanilla`, `SelfAtt`).
  * Set the embedding dimensionality `d` to match the value used in `py_code/train.py`.
  * Define the proportion of training/validation/testing datasets in the `split` argument, ensuring it aligns with the settings in `py_code/train.py`.

### Outputs:
The `evaluation()` function will return a dictionary containing TrainingLoss and TestingLoss, which store the MSE of the training and testing data, respectively.

## 3.2 Conditional Neural Process
To calculate the mean squared error (MSE) of the training and testing datasets on the Conditional Neural Process (CNP) model, please follow these steps:

1. Execute the `NPImputedAllData.py` script.
2. In line 8 of the script, specify the proportion of training/validation/testing datasets in the `split` argument. It is crucial to ensure that these proportions align with the settings used in `py_code/NPtrain.py` to maintain consistency in the dataset splits.
3. In line 10 of the script, define the appropriate data for analysis.

After running the `NPImputedAllData.py` script, the variable `error_mat_test` will store the MSE values for the all datasets.

# 4. Get estimators from PACE/1DS/MICE.
To obtain estimators from PACE, 1DS, and MICE methods, the do_analysis() function is defined in both `R_code/SimulationAnalysis.R` and `R_code/UKDataAnalysis.R` scripts. This function allows for running the analyses and retrieving the results.

### Usage:
To use the do_analysis() function, follow these examples:
* In `R_code/SimulationAnalysis.R`, lines 476, 479, and 483 demonstrate the usage of `do_analysis()` function.
* In `R_code/UKDataAnalysis.R`, lines 288 demonstrate the usage of `do_analysis()` function.

### Inputs:
* `data_name_list`: Specifies the dataset. Options:
  * In `R_code/SimulationAnalysis.R`: `data_name_list` takes a vector of data names, including `HighDim_E`, `LowDim_G`, `HighDim_G`, `LowDim_E`, `LowDim_T`, and `HighDim_T`.
  * In `R_code/UKDataAnalysis.R`: `data_name_list` should be set as `UK`.
* `dense_sparse_list`: Specifies the data type. Default values:
  * In `R_code/SimulationAnalysis.R`: `list(c("dense", "w_error"), c("sparse", "w_error"), c("dense", "wo_error"), c("sparse", "wo_error"))`.
  * In `R_code/UKDataAnalysis.R`: `c("dense", "sparse")`.
* `iidt_list`: Specifies the analysis type. Default value: `c("IID", "NonIID")`, indicating analysis for both iid and non-iid cases.
* `do_PACE`, `do_1DS`, and `do_MICE`: Specifies whether to run PACE, 1DS, and MICE methods, respectively. Default value: `TRUE` for all methods.
* `split`: Specifies the proportion of training/validation/testing data.

### Outputs:
The `do_analysis()` function returns a list result containing the MSE values for the training and testing data under different settings.

## 4.1 Using PACE and 1DS to Posteriorly Polish the Imputation from a Transformer-based Model
In `R_code/SimulationAnalysis.R`, the `do_analysis()` function is used to conduct posterior analysis using PACE and 1DS methods to refine the imputations from a Transformer-based model.

To achieve this, set `do_PACE`, `do_1DS`, and `do_MICE` to `FALSE` and `do_trans` to `TRUE`, as shown in line 479 of the code.

Please make the necessary adjustments to the function call and arguments based on your specific requirements and data.

If you have any further questions or need additional guidance, please let me know.
