"""This scripts reads files from each of the dataset and returns data in dataframe format with columns =['sentences', 'labels']
1. Both of these dataset labeling was different and designed for not only just person name extraction but also other entity like organization, location etc. 
So, we adjust these labels accourding to our required task. After processing we will have labels ["0", "B-PER", "I-PER"]

"""
import pandas as pd
from normalizer import normalize

def adjust_label(label):
    """
    Adjust labels for our task. 
    PERSON token label is converted to PER label and other labels to 'O'. As we only need person name entity token for the task.
    """
    per_label=["B-PERSON", "I-PERSON", "U-PERSON", "L-PERSON", "B-PER", "I-PER"]
    new_per_label= ["B-PER", "I-PER", "B-PER", "I-PER",  "B-PER", "I-PER"]
    ll= []
    for l in label:
        if l in per_label:
            ll.append(new_per_label[per_label.index(l)])

        else:
            ll.append('O')
    return ll


def readfile_dataset_1(file_name, do_normalize= True):
    """ Part of this code is taken from- https://github.com/Rifat1493/Bengali-NER/blob/01656468c4e6e31dd6aff4cb42be1dc751a1abcf/src/prepro.py#L5
    """
    '''
    reads file (for first dataset)
    input:
    file_name: name of the file.
    returns : DataFame containing sentence and corresponding labels
    sentences:(list of sentence) 'আশঙ্কাজনক অবস্থায় উপজেলা স্বাস্থ্য কমপ্লেক্সে নেওয়ার পথেই তাঁর মৃত্যু হয় ।',
    labels:(list of label) ['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O']
    '''
    f = open(file_name, encoding="utf-8")
    sentences = []
    sentence = []
    labels= []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n" or line[1]=="\n":
            if len(sentence) > 0:
                sentence= " ".join(sentence)
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label= []
            continue
        
        splits = line.split('\t')
        if (splits[-1]=='\n'):
            continue
        sentence.append(splits[0])
        label.append(splits[-1].split("\n")[0]) # remove extra "\n"
    
    if len(sentence) >0: # for last sentence
        sentence= " ".join(sentence)
        sentences.append(sentence)
        labels.append(label)
        sentence = []
        label= []
    
    f.close()

    df= pd.DataFrame(columns=["sentences", "labels"])
    df['sentences']= sentences
    df['labels']= labels

    if do_normalize:
        df['sentences']= df['sentences'].apply(lambda x: normalize(x))
    df['labels']= df['labels'].apply(lambda x: adjust_label(x))

    return df


def readfile_dataset_2(file_name, do_normalize= True):
    """
    Reads file of the second dataset (jsonl). 
    """
    df= pd.read_json(file_name, lines= True, encoding= 'utf-8')#
    df.rename(columns= {0:"sentences", 1: "labels"}, inplace= True)
    if do_normalize:
        df['sentences']= df['sentences'].apply(lambda x: normalize(x))
    df['labels']= df['labels'].apply(lambda x: adjust_label(x))
    
    return df


def readfile(file_name, do_normalize= True, dataset= 2):

    if dataset == 1:
        df= readfile_dataset_1(file_name= file_name, do_normalize= do_normalize)
    else:
        df = readfile_dataset_2(file_name= file_name, do_normalize= do_normalize)
    
    return df