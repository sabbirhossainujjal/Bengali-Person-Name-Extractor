# Bengali_NER
## Problem Statement
Building a person-name extractor for Bangla. It will take a sentence as input and output the person name present in the input sentence. The model should also be able to handle cases where no person’s name is present in the input sentence.

Example -
input: আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম
output: আব্দুর রহিম
input: অর্থনীতি ও আর্থসামাজিক বেশির ভাগ সূচকে বাংলাদেশ ছাড়িয়ে গেছে দক্ষিণ এশিয়াকে ।
output: 


## Solution Approach:
As this is a name entity extraction task, it was handled as token classification task. First i preprocessed the given data, making appropiate for token classification modeling, then experimented with different huggingface models for the task. Then i train the models with these experimented results and build an inference script which will load the best saved models (saved in training processe) and do prediction using the model and then post process the model output for desire output format.

## Datasets:
For this  task two dataset were used. These are open source datasets which can be downloaded from the following links.

Dataset-1: <a herf= "Bengali-NER/annotated data at master · Rifat1493/Bengali-NER">
