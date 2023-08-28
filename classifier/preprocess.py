from config import mapping
from tqdm.notebook import tqdm
import pandas as pd
from data import load_data
from datasets import load_dataset

n_labels = len(mapping)

def one_hot_encoder(df):
    one_hot_encoding = []
    for i in tqdm(range(len(df))):
        temp = [0]*n_labels
        label_indices = df.iloc[i]["labels"]
        for index in label_indices:
            temp[index] = 1
        one_hot_encoding.append(temp)
    return pd.DataFrame(one_hot_encoding)

def inspect_category_wise_data(label, n=5):
    samples = train[train[label] == 1].sample(n)
    sentiment = mapping[label]
    
    print(f"{n} samples from {sentiment} sentiment: \n")
    for text in samples["text"]:
        print(text, end='\n\n')

def load_data(dataset_loc: str):
    """Load data from source .
    """
    go_emotions = load_dataset(dataset_loc)
    data = go_emotions.data
    train, valid, test = data["train"].to_pandas(), data["validation"].to_pandas(), data["test"].to_pandas()
    return train, valid, test


train_,valid_,test_=load_data('go_emotions')

train_ohe_labels = one_hot_encoder(train_)
valid_ohe_labels = one_hot_encoder(valid_)
test_ohe_labels = one_hot_encoder(test_)


train = pd.concat([train_, train_ohe_labels], axis=1)
valid = pd.concat([valid_, valid_ohe_labels], axis=1)
test = pd.concat([test_, test_ohe_labels], axis=1)