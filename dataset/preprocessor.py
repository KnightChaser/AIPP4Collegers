import pandas as pd
from tqdm import tqdm
import os

# Define the file paths and variable names in a dictionary
datasets = {
    "krish":        "./spam_dataset_krishnamohanmaurya.csv",
    "mshe":         "./spam_dataset_mshenoda.csv",
    "shant":        "./spam_dataset_shantanudhakadd.csv",
    "enronmail":    "./spam_dataset_enron.csv",
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

# Enron company's emaile exchange. This dataset has too much word usages of "enron" itself, so we will remove the word "enron" from the dataset
enronmail["Message"] = enronmail["Subject"] + " " + enronmail["Message"]
enronmail = enronmail.drop(columns=["Subject"])
enronmail = enronmail[["Message", "Spam/Ham"]]
enronmail = enronmail.rename(columns = {"Spam/Ham": "label", "Message": "text"})
enronmail["text"] = enronmail["text"].apply(lambda x: x.replace("enron", ""))

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
merged = pd.concat([krish, mshe, shant, enronmail, jack, spamassassin, ling], ignore_index = True).progress_apply(lambda x: x.str.strip() if x.dtype == "object" else x)
merged = merged.sample(frac = 1).reset_index(drop = True).progress_apply(lambda x: x.str.strip() if x.dtype == "object" else x)
print(f"Number of rows in the dataset: {merged.shape[0]}")

exportFilePath = "./spameyes_dataset.csv"
if os.path.exists(exportFilePath):
    os.remove(exportFilePath)
merged.to_csv(exportFilePath, index = False)
print(f"Merged dataset saved as {exportFilePath}")