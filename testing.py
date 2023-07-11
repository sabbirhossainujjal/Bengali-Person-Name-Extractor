"""This scripts run testing on test dataset and returns metrics values of the models. 
This scripts is written to evaluate data as dataset-1-test format. Giving input in other format may cause fatal errors.
"""

import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from utils.configuration import CONFIG
from utils.loading_dataset import readfile
from utils.data_preprocessing import remove_erroneous_entries
from utils.training_utils import get_tokenizer, CustomDataset, Collate, NER_MODEL, testing_loop

def main():

    parser= argparse.ArgumentParser(description= "Testing for Bengali NER Task")
    parser.add_argument("--test_data_path", type= str, default= CONFIG.test_data_path, help= "Path of the test file")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide a valid huggingface language model for tokne classification", \
                        choices=["nafi-zaman/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "csebuetnlp/banglabert_large",
                                "nafi-zaman/mbert-finetuned-ner"])
    parser.add_argument("--do_normalize", type= bool, default= CONFIG.do_normalize, help= "Normalize input text or not.")
    parser.add_argument("--model_checkpoint", type= str, default= CONFIG.model_checkpoint, help= "Path to the saved model file")
    parser.add_argument("--test_batch_size", type= int, default= CONFIG.test_batch_size, help= "Test batch size")
    parser.add_argument("--max_length", type= int, default= CONFIG.max_length, help= "Maximum sequence length.")
    
    args = parser.parse_args()
    CONFIG.test_data_path= args.test_data_path
    CONFIG.model_name= args.model_name
    CONFIG.model_checkpoint= args.model_checkpoint
    CONFIG.test_batch_size= args.test_batch_size
    CONFIG.max_length= args.max_length

    test_df= readfile(file_name= CONFIG.test_data_path, do_normalize= True, dataset= 1)
    test_df= remove_erroneous_entries(test_df)
    tokenizer= get_tokenizer(model_name= CONFIG.model_name)
    collate_fn= Collate(tokenizer= tokenizer)
    test_dataset= CustomDataset(df= test_df, tokenizer= tokenizer, cfg= CONFIG)
    test_loader= DataLoader(dataset= test_dataset,
                            batch_size= CONFIG.test_batch_size,
                            collate_fn= collate_fn,
                            num_workers= CONFIG.num_workers,
                            shuffle= False,
                            pin_memory= True,
                            drop_last= False
                            )
    
    model= NER_MODEL(cfg= CONFIG)
    model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location= CONFIG.device))
    model.to(CONFIG.device)

    print(f"Running test for model {CONFIG.model_name}")
    f1_score= testing_loop(model= model, dataloader= test_loader, device= CONFIG.device)
    print(f"Model test f1_score: {f1_score}")
    print("Finished Testing.")

if __name__ == "__main__":
    main()

