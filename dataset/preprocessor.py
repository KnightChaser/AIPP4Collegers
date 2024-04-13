import pandas as pd
from tqdm import tqdm
import os

# Define the file paths and variable names in a dictionary
datasets = {
    "krish":        "./spam_dataset_krishnamohanmaurya.csv",
    "mshe":         "./spam_dataset_mshenoda.csv",
    "shant":        "./spam_dataset_shantanudhakadd.csv",
    "jack":         "./spam_dataset_jackksoncsie.csv",
    "spamassassin": "./spam_dataset_spamassassin.csv",
    "ling":         "./spam_dataset_ling.csv"
}

# Read and process the datasets using the dictionary
for alias, filepath in datasets.items():
    dataset = pd.read_csv(filepath, encoding = "latin-1")
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()
    globals()[alias] = dataset

# Unify the column names, and arrange the datasets
# @Krishnamohanmaurya's dataset
krish = krish.rename(columns = {"text_type": "label", "text": "text"})

# @Mshenoda's dataset
mshe = mshe.rename (columns = {"label": "label", "text": "text"})

# @Shantanudhakadd's dataset
shant = shant[["v1", "v2"]]
shant = shant.rename(columns = {"v1": "label", "v2": "text"})

# Jack's data has fixed "subject: " in the text, so we will remove it
jack =  jack.rename(columns = {"spam": "label", "text": "text"})
jack["label"] = jack["label"].map({0: "ham", 1: "spam"})
jack["text"] = jack["text"].str.replace("Subject: ", "")

# SpamAssassin dataset
spamassassin["body"] = spamassassin["subject"] + " " + spamassassin["body"]
spamassassin = spamassassin[["body", "label"]]
spamassassin = spamassassin.rename(columns = {"label": "label", "body": "text"})
spamassassin["label"] = spamassassin["label"].map({0: "ham", 1: "spam"})

# Ling's dataset
ling["body"] = ling["subject"] + " " + ling["body"]
ling = ling[["body", "label"]]
ling = ling.rename(columns = {"label": "label", "body": "text"})
ling["label"] = ling["label"].map({0: "ham", 1: "spam"})

# Merge the datasets with progress, and shuffle it, and export
tqdm.pandas()
merged = pd.concat([krish, mshe, shant, jack, spamassassin, ling], ignore_index = True).progress_apply(lambda x: x.str.strip() if x.dtype == "object" else x)
merged = merged.apply(lambda x: x.str.replace('\n', ' ') if x.dtype == 'object' else x)     # Remove newline characters
merged = merged[~merged['text'].str.contains('enron', case=False)]                          # Think enron emails are not appropriate for training.

print(f"Number of rows in the dataset: {merged.shape[0]}")

# Drop the duplicated rows that the "text" field is the same in the dataset
merged = merged.drop_duplicates(subset = "text", keep = False)

# Export the merged dataset
exportFilePath = "./spameyes_dataset.csv"
if os.path.exists(exportFilePath):
    os.remove(exportFilePath)
merged.to_csv(exportFilePath, index = False)
print(f"Merged dataset saved as {exportFilePath}")