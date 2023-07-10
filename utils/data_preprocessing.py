import random
import pandas as pd
from bnlp import BasicTokenizer
from sklearn.model_selection import StratifiedKFold

def check_common_entry(df1, df2):
    """
    Checks wheather the given dataset has common entries
    """
    b_tokenizer= BasicTokenizer()
    list1= [tuple(b_tokenizer.tokenize(txt)) for txt in df1['sentences']]
    list2= [tuple(b_tokenizer.tokenize(txt)) for txt in df2['sentences']]
    common= set(list1).intersection(set(list2))
    
    return common


def remove_common_entries(df, common):
    """
    removes common entries from the given dataset.
    df: from where common entrires has to remove
    common: entries that found in common. (from check_common_entry function)
    """
    common= [" ".join(txt) for txt in common]
    indexes= [df[df['sentences'] == com].index for com in common]
    index= []
    if len(indexes)>0:
        for idx in indexes:
            try:
                index.append(idx[0])
            except:
                pass
    try:
        df= df.drop(index).reset_index(drop= True)
    except:
        pass
    
    print(f"Total {len(index)} of data was removed from the given dataset.")
    
    return df


def remove_erroneous_entries(df):
    """
    Error entries are those which have unequal no of labels or words.
    
    """
    temp_df= df
    b_tokenizer= BasicTokenizer()
    temp_df['len_labels']= temp_df['labels'].apply(lambda x: len(x))
    temp_df['len_words']= temp_df['sentences'].apply(lambda x: len(b_tokenizer.tokenize(x)))
    
    error_=[]
    for i in range(len(temp_df)):
        if temp_df['len_labels'][i] != temp_df['len_words'][i]:
            error_.append(i)
    print(f"{len(error_)} no of data was detected as erroneous and discarded.")
    df= df.drop(error_).reset_index(drop= True)
    return df


def downsampling(df):
    """down-samples majority class data.
    """
    random.seed(20)
    df['per_tag']= df['labels'].apply(lambda x: 1 if x.count("B-PER") > 0 else 0)
    index_0= df[df['per_tag']==0].index # indexes of negative samples (without Name entity)
    index_1= df[df['per_tag']==1].index # indexes of positive samples (with name entity)
    index_del= None
    if len(index_0) > len(index_1):
        index = [i for i in index_0]
        index_del= random.sample(index, k= len(index_0) - len(index_1))
    elif len(index_1) > len(index_0):
        index = [i for i in index_1]
        index_del= random.sample(index, k= len(index_1) - len(index_0))
    
    if index_del is not None:
        df= df.drop(index_del).reset_index(drop= True)
    
    return df


def upsampling(df, upsample_size= 0.5):
    """upsamples minority class data.
    """
    random.seed(20)
    df['per_tag']= df['labels'].apply(lambda x: 1 if x.count("B-PER") > 0 else 0)
    index_0= df[df['per_tag']==0].index # indexes of negative samples (without Name entity)
    index_1= df[df['per_tag']==1].index # indexes of positive samples (with name entity)
    index_add= None
    
    if len(index_0) > len(index_1): 
        # upsampling class 1 
        n_diff= len(index_0) > len(index_1)
        index = [i for i in index_1]
        index_add= random.sample(index, k= int(n_diff*upsample_size) if n_diff < len(index_1) else  int(len(index_1)*upsample_size) )
    elif len(index_1) > len(index_0):
        # upsampling class 0
        n_diff= len(index_1) > len(index_0)
        index = [i for i in index_0]
        index_add= random.sample(index, k= int(n_diff*upsample_size) if n_diff < len(index_0) else  int(len(index_0)*upsample_size))

    if index_add is not None:
        temp_df= df.iloc[index_add].reset_index(drop= True)
        df= pd.concat([df, temp_df], axis= 0).reset_index(drop= True)
    
    return df


## spliting data for train and validation
def train_validation_kfold(data, n_folds= 2, seed= 20):
    data['per_tag']= data['labels'].apply(lambda x: 1 if x.count("B-PER")>0 else 0)
    skf= StratifiedKFold(n_splits= n_folds, random_state= seed, shuffle= True)

    for fold, (train_index, val_index) in enumerate(skf.split(X= data, y= data['per_tag'])):
        data.loc[val_index, 'fold']= int(fold)
        
    data['fold']= data['fold'].astype(int)
    data= data[['sentences', 'labels', 'fold']]

    return data