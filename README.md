This repository is a proof of concept for adding a Python wrapper to the
(Simple Encrypted Arithmetic Library (SEAL))[http://sealcrypto.org/], a homomorphic encryption library,
developed by researchers in the Cryp

## Main - Run with VSCode
In order to run the code using Visual Studio Code:
-   Install **Remote - Containers** extension
-   Open the folder when the code is mapped using **Remote - Containers**
-   Install all the necessary extensions in docker container
-   Run the code

## Results
| Encryption parameters:
|-- poly_modulus: 1x^4096 + 1
|-- coeff_modulus_size: 110 bits
|-- plain_modulus: 256
|-- noise_standard_deviation: 3.19

Results using the above parameters can be found in the [a relative link](results.txt) file.