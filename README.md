This repository is a proof of concept for adding a Python wrapper to the
(Simple Encrypted Arithmetic Library (SEAL))[http://sealcrypto.org/], a homomorphic encryption library,
developed by researchers in the Cryptography Research Group at Microsoft Research.

To build the wrapped Python version of SEAL, first run the executable build-docker.sh -

```
# Run in Git Bash the following command:
./build-docker.sh
```

This creates a seal package that can be imported in Python.

## Main
In order to run LogisticRegression_PySEAL example, run in PowerShell the following command -

```
# This will run the test to see if the configurations are in order:
docker run -it seal-save python3 LogisticRegression_PySEAL/test_homenc.py

# This command will run the Logistic Regression using HE and PySEAL
docker run -it seal-save python3 LogisticRegression_PySEAL/binlogreg.py
```

## VSCode
In order to run the code using Visual Studio Code:
-   Install **Remote - Containers** extension
-   Use the DockerFile to create the container
-   Run examples