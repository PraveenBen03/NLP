import os
import pandas as pd
import evaluate
import pickle
import torch

from BARTScore.bart_score import BARTScorer

import os
import argparse



from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score

from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained('klyang/MentaLLaMA-chat-7B')
model = LlamaForCausalLM.from_pretrained('klyang/MentaLLaMA-chat-7B', device_map='auto')

def load_instruction_test_data(directory_contents):
    test_data = {}
    for root in directory_contents:
      if len(root.split('.'))>1 and root.split('.')[1]=='csv':
        data = pd.read_csv(root)
        texts = data['query'].to_list()
        labels = data['gpt-3.5-turbo'].to_list()
        test_data[root.split('.')[0]] = [texts, labels]
    return test_data

def generate_response(test_data):
    generated_text = {}
    goldens = {}

    

    for dataset_name in test_data.keys():
        #if dataset_name not in ['DR', 'dreaddit']:
        #    continue
        print('Generating for dataset: {}'.format(dataset_name))
        queries, golden = test_data[dataset_name]
        goldens[dataset_name]  = golden
        responses = []

        for i in range(0, len(queries)):
            batch_data = queries[i]
            #print(batch_data[:2])
            inputs = tokenizer(batch_data, return_tensors="pt")
            #print(inputs)
            #final_input = inputs.input_ids
            generate_ids = model.generate(inputs.input_ids, max_length=2048)
            
            response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            responses.append(response)
        generated_text[dataset_name] = responses

    return generated_text, goldens

def save_output(generated_text, goldens,current_directory):
    for dataset_name in generated_text.keys():
        output = {'goldens': goldens[dataset_name], 'generated_text': generated_text[dataset_name]}
        output = pd.DataFrame(output, index=None)
        output.to_csv("{}/{}/{}.csv".format(current_directory,"save",dataset_name), index=False, escapechar='\\')

def rouge(directory_contents):
    rouge = evaluate.load('rouge')
    score_results = {}

    for root in directory_contents:
      if "otpt" in root:
        data = pd.read_csv(root)
        dname = root.split('.')[0]
        predictions = data['generated_text'].to_list()
        references = data['goldens'].to_list()
        result = rouge.compute(predictions=predictions, references=references)
        score_results[dname] = [result['rouge1'], result['rouge2'], result['rougeL']]
    return score_results

def bleu(directory_contents):
    rouge = evaluate.load('bleu')
    score_results = {}

    for root in directory_contents:
      if "otpt" in root:
        data = pd.read_csv(root)
        dname = root.split('.')[0]
        predictions = data['generated_text'].to_list()
        references = data['goldens'].to_list()
        result = rouge.compute(predictions=predictions, references=references)
        score_results[dname] = result['bleu']
    return score_results



def calculate_f1(generated, goldens):
    for dataset_name in generated.keys():
        golden = goldens[dataset_name]
        outputs = generated[dataset_name]

        output_label = []
        golden_label = []
        count = 0

        for ref, output in zip(golden, outputs):
            ref_an = ref.split("Reasoning:")[0]
            output_an = output.split("Reasoning:")[0]
            #print(output)
            #output_an = str(output)[:70]

            if dataset_name == 'swmh':
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
                    count += 1
                    output_label.append(0)
                    #print(output)

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
                elif 'no mental' in output_an.lower():
                    output_label.append(0)
                else:
                    count += 1
                    #print(output)
                    output_label.append(0)
                    #print(output)

                if 'depression' in ref_an.lower():
                    golden_label.append(2)
                elif 'suicide or self-harm' in ref_an.lower():
                    golden_label.append(1)
                elif 'ptsd' in ref_an.lower():
                    golden_label.append(3)
                elif 'no mental disorders' in ref_an.lower():
                    golden_label.append(0)

            elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
                if 'yes' in output_an.lower():
                    output_label.append(1)
                elif 'no' in output_an.lower():
                    output_label.append(0)
                else:
                    count += 1
                    output_label.append(0)
                    #print(output)

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
                elif 'decision' in output_an.lower():
                    output_label.append(7)
                elif 'other' in output_an.lower():
                    output_label.append(8)
                else:
                    count += 1
                    output_label.append(8)
                    #print(output_an)

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
                elif 'emotional turmoil' in ref_an.lower():
                    golden_label.append(6)
                elif 'everyday decision making' in ref_an.lower():
                    golden_label.append(7)
                elif 'other causes' in ref_an.lower():
                    golden_label.append(8)

            elif dataset_name == 'CAMS':
                if 'none' in output_an.lower():
                    output_label.append(0)
                elif 'bias' in output_an.lower():
                    output_label.append(1)
                elif 'jobs' in output_an.lower():
                    output_label.append(2)
                elif 'medication' in output_an.lower():
                    output_label.append(3)
                elif 'relationship' in output_an.lower():
                    output_label.append(4)
                elif 'alienation' in output_an.lower():
                    output_label.append(5)
                else:
                    count += 1
                    output_label.append(0)
                    #print(output_an)

                if 'none' in ref_an.lower():
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
        # recall = round(recall_score(f_labels, outputs, average='weighted')*100, 2)
        result = "Dataset: {}, average acc:{}, weighted F1 {}, micro F1 {}, macro F1 {}, OOD count: {}\n".format(dataset_name,
                                                                                             avg_accuracy, weighted_f1,
                                                                                             micro_f1, macro_f1, count)
        print(result)
        


# Get the current directory
current_directory = os.getcwd()

# List all elements in the current directory
directory_contents = os.listdir(current_directory)

test_data = load_instruction_test_data(directory_contents)
generated_text, goldens = generate_response(test_data)
save_output(generated_text, goldens,current_directory)
directory_contents = os.listdir(current_directory)
print(rouge(directory_contents))
print(bleu(directory_contents))
calculate_f1(generated_text, goldens)
