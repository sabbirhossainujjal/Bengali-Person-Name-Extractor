import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, AdamW, Adam
from torch.utils.data import DataLoader, Dataset

# import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer
from datasets import load_metric

from tqdm.auto import tqdm

from utils.configuration import CONFIG

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

############################################################

label_names=['O', 'B-PER', 'I-PER']
id2label= {}
label2id= {}
for i, label in enumerate(label_names):
    id2label[i]= label
    label2id[label] = i

def align_labels_with_tokens(tokens, labels):
    """
    This function aligns label to tokens. As different tokenizer use different approach for tokenizing and during tokeninzing single word can be broken into multiple
    tokens we have to align labels for these tokens. This function aligns the labels according to the tokenizer requirement.
    """
    new_labels= []
    word_ids= tokens.word_ids()
    for word_id in word_ids:
        label= -100 if word_id is None else label2id[labels[word_id]] # [CLS] has token id= 101. [PAD] token id= 0 ## tokenizer.pad_token_id
        new_labels.append(label)
    
    """-100 is used for all of the speciall token ['[CLS], ['SEP'], [PAD]'], which will help to compute loss for slot filling"""
    
    return new_labels

#############################################################
# tokenizer and dataloader

def get_tokenizer(model_name= CONFIG.model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer



class CustomDataset(Dataset):

    """Custom Dataset class. This class will convert our given text data to tokens and align the labels according to tokens and returns as a dictionary of input_ids,
    attention_mask and targets labels.
    """

    def __init__(self, df, tokenizer, cfg):
        self.df= df
        self.cfg= cfg
        self.tokenizer= tokenizer
        self.max_length= self.cfg.max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text= self.df.sentences[index]
        labels= self.df.labels[index]
        inputs= self.tokenizer(text,
                                truncation= True,
#                                add_special_tokens= True,
                                max_length= self.max_length,
                                padding= True
                                )
        
        new_labels= align_labels_with_tokens(inputs, labels)

        return {
            "input_ids": inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'targets': new_labels            
        }
    

class Collate:
    """Data collator class for creating batch with equal length of input_ids and labels which are expected for language models
    """
    def __init__(self, tokenizer):
        self.tokenizer= tokenizer
    
    def __call__(self, batch):
        output= dict()
        output["input_ids"] = [sample['input_ids'] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output['targets'] = [sample['targets'] for sample in batch]
        
        batch_max= max([len(ids) for ids in output['input_ids']])
        
        # dynamic padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [ids + (batch_max - len(ids))*[self.tokenizer.pad_token_id] for ids in output['input_ids']]
            output['attention_mask']= [mask + (batch_max - len(mask))*[0] for mask in output['attention_mask']]
            output['targets']= [target + (batch_max - len(target))*[-100] for target in output['targets']]
        else:
            output["input_ids"] = [(batch_max - len(ids))*[self.tokenizer.pad_token_id] + ids for ids in output['input_ids']]
            output['attention_mask']= [(batch_max - len(mask))*[0] + mask for mask in output['attention_mask']]
            output['targets']= [(batch_max - len(target))*[-100] + target for target in output['targets']]
        
        # convert array to tensor
        output["input_ids"] = torch.tensor(output['input_ids'], dtype= torch.long)
        output["attention_mask"] = torch.tensor(output['attention_mask'], dtype= torch.long)
        output['targets'] = torch.tensor(output['targets'], dtype=torch.long)#
        
        return output


def prepare_loader(df, tokenizer, fold, collate_fn, cfg):
    """ This function will prepare batched data from the given dataset which can be feed to language models

    Args:
        df (DataFrame): Dataset dataframe
        tokenizer: tokenizer for tokenizing
        fold (int): which fold do we want to put as validation.
        collate_fn : datacollator object
        cfg : configuration that contains all the important parameters. (CONFIG.py file)

    Returns:
        train_dataloader,
        validation_dataloader
    """
    df_train= df[df.fold != fold].reset_index(drop= True) # 2 fold out of 3 fold is used as training data, and 1 fold for validation.
    df_valid= df[df.fold == fold].reset_index(drop= True)
    # valid_labels = df_valid['labels'].values
    
    # converting dataFrame to dataset.
    train_dataset= CustomDataset(df_train, tokenizer, cfg)
    valid_dataset= CustomDataset(df_valid, tokenizer, cfg)
    
    train_loader= DataLoader(train_dataset, 
                            batch_size= cfg.train_batch_size, 
                            collate_fn= collate_fn, #merges a list of samples to form a mini-batch of Tensors
                            num_workers= cfg.num_workers, # how many subprocesses to use for data loading
                            shuffle= True, 
                            pin_memory= True,
                            drop_last= False, )
    
    valid_loader= DataLoader(valid_dataset, 
                            batch_size= cfg.valid_batch_size,
                            collate_fn= collate_fn, 
                            num_workers= cfg.num_workers,
                            shuffle= False,
                            pin_memory= True, #If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them
                            drop_last= False,
                            )
    
    return train_loader, valid_loader


################################################################
class NER_MODEL(nn.Module):
    """
    This class creates model for our name entity recognition task. It takes a language model as base model and add some other layers like dropout
    and dense layer which will returns logits for each tokens for given input.

    """
    def __init__(self, model_name= None, cfg= CONFIG):
        super(NER_MODEL, self).__init__()
        self.cfg= cfg
        self.num_labels= self.cfg.num_labels
        if model_name != None:
            self.model_name= model_name
        else:
            self.model_name= self.cfg.model_name
            
        self.model_config= AutoConfig.from_pretrained(self.model_name, output_hidden_states= True)
        self.model= AutoModel.from_pretrained(self.model_name, config= self.model_config)
        
        self.dropout= nn.Dropout(p= 0.2)
        self.linear= nn.Linear(self.model_config.hidden_size, self.num_labels)
        
    def forward(self, input_ids, attention_mask, targets= None):
        
        outputs= self.model(input_ids,
                            attention_mask= attention_mask,
                            )
        
        sequence_output= outputs[0]
#         pooled_output= outputs[1]

        entity_logits= sequence_output #self.dropout(sequence_output)
        entity_logits= self.linear(entity_logits)
        
        return entity_logits
    
##################################################
## Loss functions and metrics

metric = load_metric("seqeval")
def compute_metrics(logits, labels):
    """This function will generate f1_score and other metrics from ourt prediction and true labels

    Args:
        logits : predictions
        labels : true labels

    Returns:
        f1_score
    """
    predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[ label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return all_metrics['overall_f1']


def token_loss_fn(logits, labels, attention_mask= None):
    """Calculates crossentropy loss

    Args:
        logits : predictions
        labels : true_labels
        attention_mask ( optional): attention mask of the given input_ids. Defaults to None.
    returns:
        crossentropy loss
    """
    loss_fn= nn.CrossEntropyLoss(ignore_index= -100) # for ignoring special tokens
    num_labels= CONFIG.num_labels
    
    if attention_mask is not None:
        mask= attention_mask.view(-1) == 1 #mask for keeping the effective part
        active_logits= logits.view(-1, num_labels)[mask]
        active_labels= labels.view(-1)[mask]
#         print(active_logits.size(), active_labels.size())
        entity_loss= loss_fn(active_logits, active_labels)
        
    else:
        entity_loss= loss_fn(logits.view(-1, num_labels), labels.view(-1))
    
    return entity_loss

##################################################################
## Optimizer and scheduler

def get_optimizer(parameters, cfg= CONFIG):
    optimizer= AdamW(params=parameters, lr= cfg.learning_rate, weight_decay= cfg.weight_decay, eps= cfg.eps, betas= cfg.betas)
    return optimizer

def fetch_scheduler(optimizer, cfg= CONFIG):
    """
    Gets leanring rate schedular for given optimizer
    """
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler= lr_scheduler.CosineAnnealingLR(optimizer, T_max= cfg.T_max, eta_min= cfg.min_lr)
    elif cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler= lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0= cfg.T_0, eta_min= cfg.min_lr)
    elif cfg.scheduler== "linear":
        scheduler= lr_scheduler.LinearLR(optimizer, start_factor= 0.01, end_factor= 1.0, total_iters= 100)
    elif cfg.scheduler == None:
        return None

    return scheduler


##################################################################

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch,  cfg= CONFIG):
    """This function run one epoch of training on whole dataset.

    Args:
        model : model that is optimizing
        dataloader : train_dataloader for training
        optimizer 
        scheduler : learning rate scheduler
        epoch (int): epoch no.
        cfg: CONFIG file that contains all parameters

    Returns:
        training_loss
        f1_score
    """
    model.train()

    dataset_size= 0
    running_loss= 0.0
    score= []
    device= cfg.device
    
    progress_bar= tqdm(enumerate(dataloader), total= len(dataloader))
    steps= len(dataloader)
    for step, data in progress_bar:
        ids= data['input_ids'].to(device, dtype= torch.long)
        masks= data['attention_mask'].to(device, dtype= torch.long)
        targets= data['targets'].to(device, dtype= torch.long)
        
        batch_size= ids.size(0)
        outputs= model(ids, masks)
        loss= token_loss_fn(outputs, targets, attention_mask= masks)
        f1_score= compute_metrics(logits= outputs, labels= targets)
        score.append(f1_score)
        if cfg.gradient_accumulation_steps > 1:
            loss= loss/ cfg.gradient_accumulation_steps
        
        loss.backward()
        ## Gradient Accumulation
        if (step + 1) % cfg.gradient_accumulation_steps == 0 or step == steps:
            
            optimizer.step() #Performs a single optimization step (parameter update)
            
            # clear out the gradients of all Variables 
            # in this optimizer (i.e. W, b)
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss= running_loss/ dataset_size
        epoch_f1_score= np.mean(score)
        
        progress_bar.set_postfix(Epoch= epoch,
                                Train_loss= epoch_loss,
                                F1_Score= epoch_f1_score,
                                LR= optimizer.param_groups[0]['lr'])
    
    return epoch_loss, epoch_f1_score #

def valid_one_epoch(model, dataloader, epoch, cfg= CONFIG):
    """This function run one epoch of validation on validation dataset.

    Args:
        model : model that is optimizing
        dataloader : validation_dataloader for validate
        epoch (int): epoch no.
        cfg: CONFIG file that contains all parameters

    Returns:
        validation_loss
        f1_score
    """
    model.eval()
    
    dataset_size= 0
    running_loss= 0.0
    score= []
    device= cfg.device
    progress_bar= tqdm(enumerate(dataloader), total= len(dataloader))
    steps= len(dataloader)
    
    for step, data in progress_bar:
        ids= data['input_ids'].to(device, dtype= torch.long)
        masks= data['attention_mask'].to(device, dtype= torch.long)
        targets= data['targets'].to(device, dtype= torch.long)
        
        batch_size= ids.size(0)
        
        with torch.no_grad():
            outputs= model(ids, masks)
            loss= token_loss_fn(outputs, targets, attention_mask= masks)
            f1_score= compute_metrics(logits= outputs, labels= targets)
        
        score.append(f1_score)
        
        if cfg.gradient_accumulation_steps > 1:
            loss= loss/ cfg.gradient_accumulation_steps
        
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss= running_loss/ dataset_size
        epoch_f1_score= np.mean(score)
        
        progress_bar.set_postfix(Epoch= epoch,
                                Valid_loss= epoch_loss,
                                Valid_F1_Score= epoch_f1_score,
                                )
    return epoch_loss, epoch_f1_score #

###################################################################
import time
from collections import defaultdict

def training_loop(model, train_loader, valid_loader, optimizer, scheduler, fold, cfg= CONFIG, num_epochs= CONFIG.num_epochs, patience= 3):
    
    start= time.time()
    best_loss= np.inf
    best_score= 0
    trigger_times= 0
    history= defaultdict(list)
    
    for epoch in range(1, num_epochs+1):
        train_epoch_loss, train_f1_score= train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg= cfg)# model, dataloader, optimizer, scheduler, epoch,  cfg= CONFIG
        valid_epoch_loss, valid_f1_score = valid_one_epoch(model, valid_loader, epoch, cfg= cfg)# model, dataloader, epoch, cfg= CONFIG
        
        history['train_loss'].append(train_epoch_loss)
        history['valid_loss'].append(valid_epoch_loss)
        history['train_f1_score'].append(train_f1_score)
        history['valid_f1_score'].append(valid_f1_score)
        
        #####
        if  valid_f1_score >= best_score: #valid_epoch_loss #best_loss
            trigger_times= 0
            print(f"Vlaidation Score Improved {best_score} ---> {valid_f1_score}")
            best_score= valid_f1_score
            
            path= cfg.output_dir + f"best_model_{fold}.bin"
            torch.save(model.state_dict(), path)
            print(f"Model saved to {path}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early Stoping. \n")
                break
                
    time_elapsed= time.time() - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    
    return history, valid_epoch_loss, best_score

########################################################