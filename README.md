[This paper](http://www.aun.edu.eg/journal_files/143_J_4816.pdf) is used to develop **Naive Bayes** algorithm using federated learning.

## Run code with Visual Studio Code
-   Open the folder when the code is mapped
-   Install Python extension in order to run the script
-   Run the code

## Configurations:
The file `config.ini` can be used to change the parameters values.
Results in [results.txt](results.txt) file were obtained using local_learning and also federated_learning with the following parameters:
- n_parties: 2, 4, 6
- key_length: 1024

## Run code in Virtual Environment
- Run `./build_virtualenv.sh` file in order to create the virtual environment. **Please note that if you have MacOS/Linux machine then you need to comment line 3 and uncomment line 4 in order to actiavte the virtual environment**
- Run `./run_code.sh` file in order to run the code in the created virtual environment