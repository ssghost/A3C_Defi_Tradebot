# A3C_Defi_Tradebot
An experimental trading bot in the Defi market(Aave parachain) using A3c algorithm with Tensorflow.

### Usage
1. `$brownie compile && brownie run` for deploying all the blockchain interfaces.
2. `$python run.py` for both training and testing processes of NN model.

05.14 update: Added a multi-threads training method to A3CAgent class.
05.19 update: Multi-environments bug has been fixed.
06.01 update: Added Reliability metric method to evaluate the model performance.