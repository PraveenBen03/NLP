from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import argparse
import pandas as pd
import torch
import os

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score

def get_dict(dataset_name):
    if dataset_name == 'CAMS':
        labels = {0: 'none', 1: 'bias', 2: 'job', 3: 'medication', 4: 'relation', 5: 'alienation'}
        num_labels = 6
    elif dataset_name == 'CLP':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'DR':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'dreaddit':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'Irf':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'loneliness':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'MultiWD':
        labels = {0: 'no', 1: 'yes'}
        num_labels = 2
    elif dataset_name == 'SAD':
        labels = {0: 'school', 1: 'financial',2: 'family', 3: 'social',4: 'work', 5: 'health',6: 'emotional', 7: 'everyday',8: 'other'}
        num_labels = 9
    elif dataset_name == 'swmh':
        labels = {0: 'depression', 1: 'suicide', 2: 'anxiety', 3: 'bipolar', 4: 'no mental'}
        num_labels = 5
    elif dataset_name == 't-sid':
        labels = {0: 'depression', 1: 'suicide', 2: 'ptsd', 3: 'control'}
        num_labels = 4
    else:
        raise Exception("ERROR! please choose the correct dataset!")

    return labels, num_labels

def calculate_f1(goldens, final_labels, dataset_name):
    golden_label = []
    output_label = []
    for golden, label in zip(goldens, final_labels):
        ref_an = golden.split("Reasoning:")[0]
        output_an = label.strip()

        if 'swmh' in dataset_name:
            if 'no mental' in output_an.lower():
                output_label.append(0)
            elif 'suicide' in output_an.lower():
                output_label.append(1)
            elif 'depression' in output_an.lower():
                output_label.append(2)
            elif 'anxiety' in output_an.lower():
                output_label.append(3)
            elif 'bipolar' in output_an.lower():
                output_label.append(4)
            else:
                raise Exception('Wrong label in predictions for {}'.format(dataset_name))

            if 'no mental' in ref_an.lower():
                golden_label.append(0)
            elif 'suicide' in ref_an.lower():
                golden_label.append(1)
            elif 'depression' in ref_an.lower():
                golden_label.append(2)
            elif 'anxiety' in ref_an.lower():
                golden_label.append(3)
            elif 'bipolar' in ref_an.lower():
                golden_label.append(4)
            else:
                output_label = output_label[:-1]

        elif dataset_name == 't-sid':
            if 'depression' in output_an.lower():
                output_label.append(2)
            elif 'suicide' in output_an.lower():
                output_label.append(1)
            elif 'ptsd' in output_an.lower():
                output_label.append(3)
            elif 'control' in output_an.lower():
                output_label.append(0)
            else:
                raise Exception('Wrong label in predictions for {}'.format(dataset_name))

            if 'depression' in ref_an.lower():
                golden_label.append(2)
            elif 'suicide or self-harm' in ref_an.lower():
                golden_label.append(1)
            elif 'ptsd' in ref_an.lower():
                golden_label.append(3)
            elif 'no mental' in ref_an.lower():
                golden_label.append(0)

        elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
            if 'yes' in output_an.lower():
                output_label.append(1)
            elif 'no' in output_an.lower():
                output_label.append(0)
            else:
                raise Exception('Wrong label in predictions for {}'.format(dataset_name))

            if 'yes' in ref_an.lower():
                golden_label.append(1)
            elif 'no' in ref_an.lower():
                golden_label.append(0)

        elif dataset_name == 'SAD':
            if 'school' in output_an.lower():
                output_label.append(0)
            elif 'financial' in output_an.lower():
                output_label.append(1)
            elif 'family' in output_an.lower():
                output_label.append(2)
            elif 'social' in output_an.lower():
                output_label.append(3)
            elif 'work' in output_an.lower():
                output_label.append(4)
            elif 'health' in output_an.lower():
                output_label.append(5)
            elif 'emotional' in output_an.lower():
                output_label.append(6)
            elif 'everyday' in output_an.lower():
                output_label.append(7)
            elif 'other' in output_an.lower():
                output_label.append(8)
            else:
                raise Exception('Wrong label in predictions for {}'.format(dataset_name))

            if 'school' in ref_an.lower():
                golden_label.append(0)
            elif 'financial problem' in ref_an.lower():
                golden_label.append(1)
            elif 'family issues' in ref_an.lower():
                golden_label.append(2)
            elif 'social relationships' in ref_an.lower():
                golden_label.append(3)
            elif 'work' in ref_an.lower():
                golden_label.append(4)
            elif 'health issues' in ref_an.lower():
                golden_label.append(5)
            elif 'emotion turmoil' in ref_an.lower():
                golden_label.append(6)
            elif 'everyday decision making' in ref_an.lower():
                golden_label.append(7)
            elif 'other' in ref_an.lower():
                golden_label.append(8)

        elif dataset_name == 'CAMS':
            if 'none' in output_an.lower():
                output_label.append(0)
            elif 'bias' in output_an.lower():
                output_label.append(1)
            elif 'job' in output_an.lower():
                output_label.append(2)
            elif 'medication' in output_an.lower():
                output_label.append(3)
            elif 'relation' in output_an.lower():
                output_label.append(4)
            elif 'alienation' in output_an.lower():
                output_label.append(5)
            else:
                raise Exception('Wrong label in predictions for {}'.format(dataset_name))

            if 'no causes' in ref_an.lower():
                golden_label.append(0)
            elif 'bias or abuse' in ref_an.lower():
                golden_label.append(1)
            elif 'jobs and career' in ref_an.lower():
                golden_label.append(2)
            elif 'medication' in ref_an.lower():
                golden_label.append(3)
            elif 'relationship' in ref_an.lower():
                golden_label.append(4)
            elif 'alienation' in ref_an.lower():
                golden_label.append(5)
    avg_accuracy = round(accuracy_score(golden_label, output_label) * 100, 2)
    weighted_f1 = round(f1_score(golden_label, output_label, average='weighted') * 100, 2)
    micro_f1 = round(f1_score(golden_label, output_label, average='micro') * 100, 2)
    macro_f1 = round(f1_score(golden_label, output_label, average='macro') * 100, 2)
    print("Dataset: {}, average acc:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(dataset_name,
                                                                                         avg_accuracy, weighted_f1,
                                                                                         micro_f1, macro_f1))

gen_directory = os.getcwd()



curd = "\save"


    
data_directory = gen_directory + curd

files_in_directory = os.listdir(data_directory)
cal_data = {}
for root in files_in_directory : 
    data = pd.read_csv(os.path.join(data_directory, root))
    goldens = data['goldens'].to_list()
    generated_text = data['generated_text'].to_list()

    dataset_name = root.split('.')[0]
    labels, num_labels = get_dict(dataset_name)
    print('Generating for {} dataset.'.format(dataset_name))

    mentalbert = BertForSequenceClassification.from_pretrained(os.path.join(gen_directory+"\Models",dataset_name), num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(gen_directory+"\Models",dataset_name))

    all_labels = []
    for i in range(0, len(generated_text)):
        batch_data = generated_text[i]
        inputs = tokenizer(batch_data, return_tensors="pt", truncation=True, max_length=512)
        outputs = mentalbert(**inputs)[0]
        outputs = outputs.cpu().detach().numpy()
        all_labels.append(np.argmax(outputs))
    final_labels = []
    for num in all_labels:
        final_labels.append(labels[num])
    cal_data[dataset_name] = [goldens, final_labels]       
for dataset_name in cal_data.keys():
    goldens, final_labels = cal_data[dataset_name]
    calculate_f1(goldens, final_labels, dataset_name)




