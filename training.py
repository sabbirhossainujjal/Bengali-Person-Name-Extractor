import pandas as pd
import numpy as np
import argparse
from normalizer import normalize #git+https://github.com/csebuetnlp/normalizer

from utils.configuration import CONFIG
from utils.loading_dataset import readfile
from utils.data_preprocessing import remove_erroneous_entries, upsampling, downsampling, train_validation_kfold
from utils.training_utils import get_tokenizer, Collate, prepare_loader, NER_MODEL, get_optimizer, fetch_scheduler, training_loop

def main():
    ##Run full training scripts from data loading to training steps.

    # parse the args
    parser= argparse.ArgumentParser(description= "Training for Bengali NER Task")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide a valid huggingface language model for tokne classification", \
                        choices=["nafi-zaman/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "csebuetnlp/banglabert_large",
                                "nafi-zaman/mbert-finetuned-ner"])
    parser.add_argument("--do_normalize", type= bool, default= CONFIG.do_normalize, help= "Normalize input text or not.")
    parser.add_argument("--do_downsampling", type= bool, default= CONFIG.do_downsampling, help= "Do downsample majority class or not.")
    parser.add_argument("--do_upsampling", type= bool, default= CONFIG.do_upsampling, help= "Do upsample minority class or not.")
    parser.add_argument("--upsample_size", type= float, default= CONFIG.upsample_size, help= "Percent of data to be upsampled.")
    parser.add_argument("--output_dir", type= str, default= CONFIG.output_dir, help= "Path to the output model files")
    parser.add_argument("--debug", type= bool, default= CONFIG.debug, help= "Runing mode [debug or full trainig]")
    parser.add_argument("--n_folds", type= int, default= CONFIG.n_folds, help= "Number of fold to train")
    parser.add_argument("--num_epochs", type= int, default= CONFIG.num_epochs, help= "Number of epochs to run")
    parser.add_argument("--train_batch_size", type= int, default= CONFIG.train_batch_size, help= "Training batch size")
    parser.add_argument("--valid_batch_size", type= int, default= CONFIG.valid_batch_size, help= "Validation batch size")
    parser.add_argument("--gradient_accumulation_steps", type= int, default= CONFIG.gradient_accumulation_steps, help= "Gradient Accumulation Steps")
    parser.add_argument("--learning_rate", type= float, default= CONFIG.learning_rate, help= "Initial Learning Rate")
    parser.add_argument("--scheduler", type= str, default= CONFIG.scheduler, help="Learning rate scheduler.",
                        choices=["CosineAnnealingWarmRestarts", "CosineAnnealingLR", "linear"])
    parser.add_argument("--max_length", type= int, default= CONFIG.max_length, help= "Maximum sequence length.")
    
    args = parser.parse_args()
    CONFIG.model_name= args.model_name
    CONFIG.output_dir= args.output_dir
    CONFIG.do_normalize= args.do_normalize
    CONFIG.do_downsampling= args.do_downsampling
    CONFIG.do_upsampling= args.do_upsampling
    CONFIG.upsample_size= args.upsample_size
    CONFIG.debug= args.debug
    CONFIG.n_folds= args.n_folds
    CONFIG.num_epochs= args.num_epochs
    CONFIG.train_batch_size= args.train_batch_size
    CONFIG.valid_batch_size= args.valid_batch_size
    CONFIG.gradient_accumulation_steps= args.gradient_accumulation_steps
    CONFIG.learning_rate= args.learning_rate
    CONFIG.scheduler= args.scheduler
    CONFIG.max_length= args.max_length

    #loading datasets
    dataset1_train= readfile(file_name= CONFIG.data_path1_train, do_normalize= CONFIG.do_normalize, dataset= 1)
    dataset2= readfile(file_name= CONFIG.data_path2, do_normalize= CONFIG.do_normalize, dataset= 2)
    dataset1_train= remove_erroneous_entries(dataset1_train)
    dataset2= remove_erroneous_entries(dataset2)

    ## As datasets are small in size we concat dataset 1 and dataset 2 for training.
    dataset= pd.concat([dataset2, dataset1_train], axis= 0).reset_index(drop= True)

    if CONFIG.do_upsampling:
        dataset= upsampling(df= dataset, upsample_size= 0.5)
    if CONFIG.do_downsampling:
        dataset= downsampling(df= dataset) ## Down-sampling of majority class data to prevent overfitting.

    # taking a small datset for debuging
    if CONFIG.debug:
        data= dataset[['sentences', 'labels']][: CONFIG.dataset_size]
    else:
        data= dataset[['sentences', 'labels']]
    
    ## building kfold data of the dataset
    data = train_validation_kfold(data= data, n_folds= CONFIG.n_folds, seed= CONFIG.seed)

    # loading tokenizer and collate function
    tokenizer= get_tokenizer(model_name= CONFIG.model_name)
    collate_fn= Collate(tokenizer= tokenizer)

    fold_scores= []
    for fold in range(CONFIG.n_folds):
        print(f"====== Started Training Fold-{fold} ======")

        ## loading necessary data loader and model from trianing_utils.py
        train_loader, valid_loader= prepare_loader(df= data, tokenizer= tokenizer, fold= fold, collate_fn= collate_fn, cfg= CONFIG)
        model= NER_MODEL(cfg= CONFIG)
        model.to(device= CONFIG.device)
        optimizer= get_optimizer(model.parameters(), cfg= CONFIG)
        scheduler= fetch_scheduler(optimizer= optimizer)

        # run training
        history, epoch_loss, f1_score= training_loop(model,  train_loader, valid_loader, optimizer, scheduler, fold= fold, cfg= CONFIG, num_epochs= CONFIG.num_epochs, patience= CONFIG.patience)
        print("\n\n")
        print(f"Fold [{fold}] avg loss: {epoch_loss}\n")
        print(f"Fold [{fold}] avg score: {f1_score}\n")
        fold_scores.append(f1_score)
        
        if fold < CONFIG.n_folds-1:
            del model
        del train_loader, valid_loader

    print("====== xxxx ======")
    print(f"Overall score: {np.mean(np.mean(fold_scores, axis= 0))}")
    print(f"====== xxxx ======")


if __name__ == "__main__":
    main()