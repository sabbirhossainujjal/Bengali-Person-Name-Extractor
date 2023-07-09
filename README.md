# Bengali_NER
## Problem Statement
Building a person-name extractor for Bangla. It will take a sentence as input and output the person name present in the input sentence. The model should also be able to handle cases where no person’s name is present in the input sentence.

Example -
<br>input: আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম
<br>output: আব্দুর রহিম
<br>input: অর্থনীতি ও আর্থসামাজিক বেশির ভাগ সূচকে বাংলাদেশ ছাড়িয়ে গেছে দক্ষিণ এশিয়াকে ।
<br>output: [] 


## Solution Approach
As this is a name entity extraction task, it was handled as token classification task. First i preprocessed the given data, making appropiate for token classification modeling, then experimented with different huggingface models for the task. Then i train the models with these experimented results and build an inference script which will load the best saved models (saved in training processe) and do prediction using the model and then post process the model output for desire output format.

## Datasets
For this  task two dataset were used. These are open source datasets which can be downloaded from the following links.

Dataset-1: <a href= "https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data"> [Rifat1493/Bengali-NER] </a>
<br> Dataset-2: <a href= "https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl"> [banglakit/bengali-ner-data] </a>

### Dataset-1 description: 
Dataset -1 contains annotation data in `.txt` file format. From this dataset repository we take train_data.txt and test_data.txt file for our task. These files resides in <a href= "https://github.com/Rifat1493/Bengali-NER/blob/master/Input/train_data.txt"> master/inptut/train_data.txt </a> and <a href= "https://github.com/Rifat1493/Bengali-NER/blob/master/Input/test_data.txt">master/inptut/test_data.txt </a>.

<br>`Total sentences in train.txt= 4612`
<br>`Total sentences in test.txt= 1950`

<br>Annotation format in Dataset-1:
<br>লালপুর	B-LOC
<br>(	O
<br>নাটোর	B-LOC
<br>)	O
<br>প্রতিনিধি	O
<br>ব্রাহ্মণবাড়িয়া-২	B-LOC
<br>(	O
<br>সরাইল-আশুগঞ্জ	I-LOC
<br>)	O
<br>আসনে	O
<br>নির্বাচন	O
<br>থেকে	O
<br>সরে	O
<br>দাঁড়িয়েছেন	O
<br>আওয়ামী	B-ORG
<br>লীগের	I-ORG
<br>নেতৃত্বাধীন	O
<br>১৪-দলীয়	B-ORG
<br>জোটের	I-ORG
<br>শরিক	O
<br>জাসদের	B-ORG
<br>(	O
<br>ইনু	B-PER
<br>)	O
<br>প্রার্থী	O
<br>আবু	I-PER
<br>বকর	I-PER
<br>মো	B-PER
<br>.	I-PER
<br>ফিরোজ	I-PER
<br>।	O

### Dataset-2 description
Dataset-2 contains data in `.jsonl` file format. This dataset is arranged in one file resides in <a href="https://github.com/banglakit/bengali-ner-data/blob/master/main.jsonl">master/main.jsonl</a>

<br>`Total sentences in dataset-2 : 3545`
<br>Annotation format in dataset-2:
<br>["মো. নাহিদ হুসাইন নামের এক পরীক্ষার্থী অভিযোগ করেন, ইডেন মহিলা কলেজের পাঠাগার ভবনের দ্বিতীয় তলায় তাঁর পরীক্ষার আসন ছিল।", ["B-PERSON", "I-PERSON", "L-PERSON", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "L-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O"]]


#### Dataset disribution:

![1](Screenshots/dataset_distribution.png)

### Dataset Preprocessing

#### Dataset loading and annotation
From the previous description we have seen that the two dataset are in two different format and their annotation format is totally different. So we have to pre-processe these data to convert these annotation to a common format. Moreover, there are some redundant annotations in the datasets like "ORG", "LOC" etc. We dont need these annotations. 
<br> First we converted all the input text in a common format. Our intented format is sentence level i.e all tokens of a sentence are combined in sentence not tokenized like in dataset-1. For that we converted dataset-1 to out intented format.

<br> For our person name extraction task we only need `B-PER`(Begining of the name), `I-PER`(Inside of the name), `O`(others). So we converted all the annotations in both dataset to our desired format.

<br> Example:
<br>`previous annotation`: ["B-PERSON", "I-PERSON", "L-PERSON", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "L-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
<br>`new_annotation`: ["B-PER", "I-PER", "L-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]]

Coding for data-loading and pre-processing is implemented in `loading_dataset.py`. After running this script we will have a dataframe format data for particular dataset which can be used later for training purpose.

#### Exploratory Data Analysis (EDA)
After loading our data we did some data analysis. First we check all the all the data have proper annotation, especially we checked every token in the given text have corresponding annotaion. If there are missing annotation in the dataset, this data can't be used in training. 
<br> After analysing we found `1019` miss-match data in dataset-1 and `172` mis-match data in dataset-2. So we discarded these erroneous data from our datasets. 
<br> Then we checked if there is any common entries in train test datasets. For our test dataset we used dataset-1 testing data. After checking we found `67` common entries in train, test dataset. So we removed these common data from test dataset, because these common data may introduce data-leakage problem in our experiment.

<br> Next we check the distribution of our datasets. We checked how many input entries contains name entity. From my investigation, i found that there was not many name annotation data in the datasets. There were a huge imbalance in the datasets. The distribution of these datasets are given bellow.

![2]("Screenshots/11.jpg")

<br> From the distribution we found that there are not much data with person token so, we have combined both of these dataset to a single dataset for training. And to metigate data imbalance problem we `downsample` data without person token sothat it doesn't suffer overfitting problem.
<br> After combining both of the data and downsampling majority class our dataset looks like the following ditribution.
![3](Screenshots/combined_data_distribution.png)

#### Aligning labels to tokens

In language model we convert text token to a particular number so that our model can process the data. As different model uses different tokenization scheme like word toekenizer, sub-word tokenizer etc. In sub-word tokenizer any token may broken into multiple tokens before converting into corresponding numerical value. For that there may be incoherence in tokens labels as our tokens are word level. So we implemented a fuction which will align our old lebels to new label that match the tokenized tokens.

> All these preprocessing functions will be found in `utils/data_preprocessing.py` script.

#### Train-validation split
For train-validation split, I have used stratifiedkfold from sklearn. Here i used this split method so that our train valid dataset have properly distributed samples from both class (with person token and without person token)


### Modeling

#### Models
For modeling we used bert-based huggingface models. We used 3 models in our experiments. 
1. <a href= "https://huggingface.co/nafi-zaman/celloscope-28000-ner-banglabert-finetuned">banglabert finetuned for NER task
2. <a href= "https://huggingface.co/csebuetnlp/banglabert">base banglabert
3. <a href= "https://huggingface.co/nafi-zaman/mbert-finetuned-ner">mbert finetuned for NER task

We build a custom model class which will load the desire model and add some final dense layers for entity classification task.

#### Loss function and metrices
For our task we use `CrossEntropyLoss` loss function for calculating our model loss. For monitoring our model performance we used F1_score as model metric.

#### Optimizer and scheduler
For optimizer we used `AdamW` optimizer and for learning rate scheudler we tried three types of scheduler -`[CosineAnnealingLR, CosineAnnealingWarmRestarts, linear]`

### Training
We build custom `training loop` for training and validating our model performance. We trained each of the model and done `3 fold cross-validation`. Performance of these models are listed in the following table.

