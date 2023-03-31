import os
import gc
import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments
from transformers import Trainer

class Seq2TagTrainer(object):
    """
    Trains a Ukrainian Seq2Tag GEC model with the parameters passed.
    """
    def __init__(self, train_path, dev_path, model_name, model_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        Initializing the tokenizer and the baseline model.
        Usable options
        - https://huggingface.co/ukr-models/xlm-roberta-base-uk
        - https://huggingface.co/xlm-roberta-base
        - https://huggingface.co/youscan/ukr-roberta-base
        """
        self.tokenizer = AutoTokenizer.from_pretrained("youscan/ukr-roberta-base", add_prefix_space=True)
        
    def setup(self):
        """
        Sets the basics up.
        """
        # creating the output folder
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # Writing model metadata
        # Write train metadata
        message = f"My name is {self.model_name}\n"
        message += "Train datetime: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        message += "\n"

        #we use append here in case we fine-tune an existing model
        with open(self.model_path + "/metadata.txt", 'a') as metadata_file:
            metadata_file.write(message)
    
    # takes in Dataset hugging face object, tokenizes words into wordpieces, aligns lables with tokenized input"
    def tokenize_and_align_labels(self, examples):
        global label_encoding_dict
        label_all_tokens = True
        tokenized_inputs = self.tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif label[word_idx] == '0':
                    label_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_encoding_dict[label[word_idx]])
                else:
                    label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def load_data(self):
        """
        Loads the data and prepares the datasets.        
        """
        global label_encoding_dict
        # reading the input data
        with open(self.train_path, 'r') as f:
            train_json = json.load(f)

        with open(self.dev_path, 'r') as f:
            dev_json = json.load(f)

        # converting it to pandas dataframe first
        train_df = pd.DataFrame(train_json)
        dev_df = pd.DataFrame(dev_json)

        # assigning the column names
        train_df.columns = ['tokens', 'labels']
        dev_df.columns = ['tokens', 'labels']

        # converting both sets in the required format
        train_dataset = Dataset.from_pandas(train_df)
        dev_dataset = Dataset.from_pandas(dev_df)

        #Generate label vocab from the dataset
        all_labels = [label for sequence in list(train_df['labels']) for label in sequence]
        label_list = list(set(all_labels))

        # generate label_encoding_dict for use in tokenization
        i = 0
        label_encoding_dict = {}
        for label in label_list:
            label_encoding_dict[label] = i
            i += 1

        # save label_encodings for later use in the model folder
        lines = [f'{label} {label_encoding_dict[label]}\n' for label in label_encoding_dict]
        with open(f'{model_path}/label_encoding.txt', 'w') as label_file:
            label_file.writelines(lines)

        train_tokenized_datasets = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        dev_tokenized_datasets = dev_dataset.map(self.tokenize_and_align_labels, batched=True)
        return train_tokenized_datasets, dev_tokenized_datasets, label_list

    # Defines metrics to be computed. Adapted from old eval
    def compute_metrics(p):
        global label_list
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        # Dataset-wide metrics
        total = 0
        TN = 0
        FN = 0
        TP = 0
        FP = 0

        # types of FP
        # extra_error + bad_guess = FP
        extra_error = 0
        bad_guess = 0

        # Classifying label combinations
        for sent_idx in range(len(true_labels)):
            for tag_idx in range(len(true_labels[sent_idx])):
                model_label = predictions[sent_idx][tag_idx]
                real_label = true_labels[sent_idx][tag_idx]
                if (real_label == '$KEEP' and model_label == '$KEEP'):
                    TN += 1
                elif real_label != '$KEEP' and model_label == '$KEEP':
                    FN += 1
                elif real_label == '$KEEP' and model_label != '$KEEP':
                    FP += 1
                    extra_error += 1
                if real_label !='$KEEP' and model_label != '$KEEP':
                    if model_label == real_label:
                        TP += 1
                    if model_label != real_label:
                        FP += 1
                        bad_guess += 1

        total = FP+TP+FN+TN
        print(f'TP:{TP}')
        print(f'FP:{FP}')
        print(f'TN:{TN}')
        print(f'FN:{FN}')
        print(f'total:{total}')
        print()

        # Calculating metrics
        
        # Accuracy     
        accuracy = (TP+TN)/total
        # Precision     
        if (TP+FP) != 0:
            precision = TP/(TP+FP)
        else:
            precision = 'n/a'
        # Recall
        if (TP+FN) != 0:
            recall = TP/(TP+FN)
        else:
            recall = 'n/a'
        # F score
        if recall != 'n/a' and precision != 'n/a':
            f1 = 2*(recall*precision)/(recall+precision)
            fhalf = (1.25*precision*recall)/(0.25*precision + recall)
        else:
            f1 = 'n/a'
            fhalf = 'n/a'

        # Bad guess/extra error
        if FP != 0:
            extra_error_proportion = extra_error/FP
            bad_guess_proportion = bad_guess/FP
        else:
            extra_error_proportion = 'n/a'
            bad_guess_proportion  = 'n/a'

        out = {"accuracy": accuracy,
            "precision": precision, 
            "recall": recall, 
            "f1": f1,
            "f0.5":fhalf,     
            "extra-error":extra_error_proportion,
            "bad-guess":bad_guess_proportion}

        # print('\n\n')
        # for key in out:
        #   print(f'{key}: {round(out[key], 3)}')
        # print('\n\n')
        # ! FIXME ↑↑↑
        # ! type str doesn't define __round__ method

        # for now, dont print exrtra-error and bad-guess. 
        #will be included in next iteration of compute metrics (together with label_data)
        out.pop('extra-error')
        out.pop("bad-guess")

        return out

    def train(self):
        """
        Trains and saves the model.
        """
        self.setup()
        global label_list
        train_tokenized_datasets, dev_tokenized_datasets, label_list = self.load_data()
        # Initializing the model. The baseline should match the tokenizer!
        model = AutoModelForTokenClassification.from_pretrained("youscan/ukr-roberta-base", num_labels=len(label_list))
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer) # initialize data collator
        callback = EarlyStoppingCallback(early_stopping_patience=1)

        # initialize training args
        args = TrainingArguments(
            self.model_path,
            evaluation_strategy = "epoch", 
            save_strategy = "epoch",
            load_best_model_at_end=True,
            gradient_accumulation_steps = 4,
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=1e-5,
        )

        # cleaning the memory
        gc.collect()

        trainer = Trainer(
            model,
            args,
            callbacks=[callback],
            train_dataset=train_tokenized_datasets,
            eval_dataset=dev_tokenized_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.evaluate()
        trainer.save_model('model')
        print("Done!")