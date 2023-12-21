# Ethereum Address Clustering Analysis

## Overview
This Python script performs an analysis and clustering of Ethereum addresses based on their on-chain transaction activity. The script connects to the Ethereum blockchain, retrieves recent transactions, and applies machine learning techniques to cluster addresses with similar characteristics.

## Requirements
- Python 3
- Libraries: `web3`, `pandas`, `requests`, `scikit-learn`, `matplotlib`, `seaborn`
- Ethereum node access (via Infura)
- Etherscan API key

## Functionality
1. **Ethereum Connection**: Connects to the Ethereum blockchain using Infura to fetch recent transaction data.

2. **Data Collection**:
   - Collects the latest 100 unique Ethereum addresses from the recent blocks.
   - Fetches the balance of these addresses.

3. **Transaction History**:
   - Retrieves the transaction history for each address with a balance of at least 1 ETH.
   - Uses the Etherscan API for detailed transaction data.

4. **Data Aggregation**:
   - Aggregates transaction data for each address, calculating mean values for attributes like `value`, `gas`, and `gasPrice`.

5. **Data Preprocessing**:
   - Detects and handles categorical and numerical columns.
   - Applies standard scaling to numerical columns and one-hot encoding to categorical columns.

6. **Clustering**:
   - Performs clustering using K-Means on preprocessed data.
   - The number of clusters is set to 5, with `n_init` parameter explicitly specified for stability.

7. **Dimensionality Reduction and Visualization**:
   - Applies TruncatedSVD for dimensionality reduction to visualize clusters.
   - Generates a scatter plot showing the clusters, saved as a PNG file.

## Usage
1. Set up your Python environment and install the required libraries.
2. Insert your Infura URL and Etherscan API key in the script.
3. Run the script to perform the analysis. The final scatter plot will be saved as `cluster_visualization.png`.

## Notes
- Ensure that your Infura and Etherscan keys are kept secure.
- The script may need modifications based on the latest Ethereum blockchain changes or API updates.
