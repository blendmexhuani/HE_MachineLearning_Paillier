**Python-Paillier** implementation of naive bayes algorithm using federated learning with Iris, Wine and Breast Cancer datasets from sklearn.
[This paper](http://www.aun.edu.eg/journal_files/143_J_4816.pdf) is used for the development of the algorithm using federated learning.

## Configurations
The file `config.ini` can be used to change the parameters values.
Results in [results.txt](results.txt) file were obtained using local_learning and also federated_learning with the following parameters:
- n_parties: 1, 2, 4, 6
- key_length: 1024

**Do not set the n_parties greater than 7 because of the size of the datasets.**

## Run code using Virtual Environment for Python
If you have Windows installed, you first need to:
- Type `regedit` in the Windows start menu to launch Registry Editor
- Go to the `Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
- Edit the value of the `LongPathsEnabled` property and set it to **1**

After setting that or you have MacOS/Linux installed then:
- Run `./build_virtualenv.sh` file in order to create the virtual environment. **Please note that if you have MacOS/Linux then you need to comment line 3 and uncomment line 4 inside `build_virtualenv.sh` file in order to activate the virtual environment**
- Run `./run_code.sh` file in order to run the code inside the created virtual environment. **Please note that if you have MacOS/Linux then you need to comment line 1 and uncomment line 2 inside `run_code.sh` file in order to activate the virtual environment**

## Run code with Visual Studio Code
-   Open the folder when the code is mapped
-   Install Python extension
-   Run the code
