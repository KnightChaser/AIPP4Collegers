import pandas as pd
import os

# Define the file paths and variable names in a dictionary
datasets = {
    "krish":    "./spam_dataset_krishnamohanmaurya.csv",
    "mshe":     "./spam_dataset_mshenoda.csv",
    "shant":    "./spam_dataset_shantanudhakadd.csv"
}

# Read and process the datasets using the dictionary
for alias, filepath in datasets.items():
    dataset = pd.read_csv(filepath, encoding = "latin-1")
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()
    globals()[alias] = dataset

# Unify the column names
krish = krish.rename(columns = {"text_type": "label", "text": "text"})
mshe  = mshe.rename (columns = {"label": "label", "text": "text"})
shant = shant[["v1", "v2"]]
shant = shant.rename(columns = {"v1": "label", "v2": "text"})

# Merge the datasets
merged = pd.concat([krish, mshe, shant], ignore_index = True)
print(merged)

# Save the merged dataset as a CSV file
exportFilePath = "./spameyes_dataset.csv"
if os.path.exists(exportFilePath):
    os.remove(exportFilePath)
merged.to_csv(exportFilePath, index = False)
print(f"Merged dataset saved as {exportFilePath}")