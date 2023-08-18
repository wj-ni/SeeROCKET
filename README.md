# SeeRocket

Source codes for **SEE**ing predicting by **R**and**O**m **C**onvolutional **KE**rnel **T**ransform

# Requirement

- Python, NumPy, pandas
- Numba(0.50+)
- scikit-learn， lightgbm
- joblib, scipy

# Code Structure

seerocket.py: Implementation of feature transformation in SeeRocket

seerocket_multivariate.py: Expansion of feature transformation in SeeRocket on multivariate variables

LightGBM.py: Example of time series prediction using LightGBM in prediction models

# Usage

## Example command

`python LightGBM.py --config ../config.ini`

In this example, the --config flag is used to specify the path to the configuration file, which is ../config.ini. 

## Example Configuration File

```
[run_params]
minibatch_size: 3000
train_file_path = ../dataset/seeing_train.csv
test_file_path: ../dataset/seeing_test.csv
save_path: ../result
target_column: ccd_avg
time_column: time
sequence_length: 32

[seerocket_params]
num_features = 10000
max_dilations_per_kernel = 32
max_allow_num_channels = 9
kernel_size = 9
num_ones = 1
dilations = None
initial_stride = 1
stride_increment = 0
LPPV_size = 1
padding_left = 0
padding_right = 0
padding_value = 0
num_kernels = None

[LightGBM_params]
objective = regression
boosting_type = gbdt
force_col_wise = True
max_depth = 5
num_leaves = 22
learning_rate = 0.06698
```

## Configuration File Details

```
[run_params]
minibatch_size              The size of each mini-batch
train_file_path             The path to the training data file
test_file_path              The path to the test data file
save_path                   The path to save the model and results
target_column               The name of the target column
time_column                 The name of the time column
sequence_length             The length of the time series

[seerocket_params]
num_features                The number of features
max_dilations_per_kernel    The maximum number of dilations per kernel
max_allow_num_channels      The maximum number of used channels
kernel_size                 The length of the kernel
num_ones                    The number of ones in each kernel
dilations                   Dilation coefficients, can be a list or None
initial_stride              The initial stride value
stride_increment            The increment value for stride
LPPV_size                   The size of LPPV
padding_left                The padding length on the left side
padding_right               The padding length on the right side
padding_value               The value used for padding, default is 0
num_kernels                 The number of kernels, default is None

[LightGBM_params]
objective                   Objective function type, here is regression
boosting_type               Boosting type, here is gbdt
force_col_wise              Force to use column-wise optimization method
max_depth                   Maximum depth of the tree
num_leaves                  Number of leaves on the tree
learning_rate               Learning rate
```

