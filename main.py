from web3 import Web3
import pandas as pd
import requests
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

# Connect to Ethereum using Infura
infura_url = 'https://mainnet.infura.io/v3/64331e02e85c40cd8ac13d8654338f21'
# Etherscan API Key
ETHERSCAN_API_KEY = "GW8D1S6DCM1ZNFEWURP47PBJPBYCXXK4HI"
web3 = Web3(Web3.HTTPProvider(infura_url))

# Fetch the latest block number
latest_block = web3.eth.block_number

# Container for unique addresses
unique_addresses = set()

# Number of blocks to check (adjust as needed)
num_blocks_to_check = 1000  # This is an arbitrary number for demonstration

for i in range(num_blocks_to_check):
    # Fetch the block
    block = web3.eth.get_block(latest_block - i, full_transactions=True)

    # Iterate through transactions in the block
    for tx in block.transactions:
        unique_addresses.add(tx['from'])  # Add sender address
        if 'to' in tx and tx['to']:       # Add receiver address, if present
            unique_addresses.add(tx['to'])

    # Break if we've collected enough addresses
    if len(unique_addresses) >= 100:
        break

# Convert to list and get the most recent 10,000 addresses
recent_addresses = list(unique_addresses)[:100]

# Prepare data for DataFrame
data = []

for address in recent_addresses:
    balance = web3.eth.get_balance(address)
    eth_balance = web3.from_wei(balance, 'ether')

    # Example: Fetching the latest transaction count (nonce) for each address
    #nonce = web3.eth.get_transaction_count(address)

    # Append data (you can add more details similarly)
    data.append({
        'address': address,
        'balance': eth_balance,
        #'nonce': nonce,
        # Add more details as needed
    })

# Create DataFrame
df = pd.DataFrame(data)

df_filtered = df[df['balance'] >= 1]

# List of addresses
addresses = df_filtered["address"]

# Function to get transaction history
def get_transaction_history(address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data['result']

# DataFrame to store all transactions
all_transactions = pd.DataFrame()

# Fetch transactions for each address
for address in addresses:
    transactions = get_transaction_history(address)
    if transactions:
        df = pd.DataFrame(transactions)
        all_transactions = pd.concat([all_transactions, df], ignore_index=True)
    time.sleep(0.5)


df = all_transactions

# Function to detect categorical columns
def detect_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# Function to flatten multi-level column names
def flatten_columns(df):
    df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]
    return df

# Preprocessing and aggregation
features = df.groupby('from').agg({
    'value': 'mean',
    'gas': 'mean',
    'gasPrice': 'mean',
    # Add more features
}).reset_index()

# Flatten the columns
features = flatten_columns(features)

# Detecting categorical and numerical columns
categorical_cols = detect_categorical_columns(features)
numerical_cols = [col for col in features.columns if col not in categorical_cols and col != 'from']

# Preprocessor for pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Clustering pipeline with explicit n_init
cluster_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('cluster', KMeans(n_clusters=5, n_init=10))])

# Apply clustering
cluster_pipeline.fit(features)
clusters = cluster_pipeline['cluster'].labels_

# Add cluster labels to the DataFrame
features['cluster'] = clusters

# Dimensionality Reduction for Visualization
# Ensure only numerical columns are used for PCA
pca_features = preprocessor.transform(features)
svd = TruncatedSVD(n_components=2)
reduced_features = svd.fit_transform(pca_features)

# Create a DataFrame for the reduced features
reduced_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
reduced_df['cluster'] = features['cluster']

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='cluster', palette='viridis', alpha=0.7)
plt.title('Cluster Visualization')
plt.savefig('cluster_visualization.png', dpi=300)  # Save as PNG with high resolution
