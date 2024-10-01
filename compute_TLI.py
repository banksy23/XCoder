import json
from nltk.util import ngrams
import argparse
from tqdm import tqdm
import pdb
import numpy
import heapq

def read_jsonl(file_path):
    """Reads a JSON Lines file and returns the data from each line."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def write_jsonl(file_path, data):
    """Writes data to a JSON Lines file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_texts(item, keys):
    """Extracts text from JSON Lines formatted data."""
    if 'message' in keys or 'messages' in keys:
        assert len(keys) == 1
        key = keys[0]
        try:
            return ' '.join(message['content'] if message['role'].lower() == 'user' else '' for message in item[key])
        except:
            pdb.set_trace()
    else:
        return ' '.join([item[k] for k in keys])

def generate_grams(data, keys, ngram_sizes):
    """Generates n-grams and stores them in data['grams']"""
    for item in data:
        text = extract_texts(item, keys)
        grams = set()
        for n in ngram_sizes:
            grams.update(ngrams(text.split(), n, pad_left=True, pad_right=True, left_pad_symbol='<start>', right_pad_symbol='<end>'))
        item['grams'] = grams

def calculate_similarity(train_data, test_data, args):
    generate_grams(train_data, args.key_train, args.ngram_sizes)
    generate_grams(test_data, args.key_test, args.ngram_sizes)

    top = args.top

    for train_data_item in train_data:
        train_data_item['max_similarity'] = 0  # Ensuring this is initialized

    for test_data_item in tqdm(test_data):
        heap = []  # Min-heap to store top similarities
        test_grams_size = len(test_data_item['grams'])

        for i, train_data_item in enumerate(train_data):
            repeat_grams = train_data_item['grams'] & test_data_item['grams']
            similarity = len(repeat_grams) / test_grams_size

            if 'max_similarity' not in train_data_item or train_data_item['max_similarity'] < similarity:
                train_data_item['max_similarity'] = similarity

            if len(heap) < top:
                heapq.heappush(heap, (similarity, i))
            else:
                heapq.heappushpop(heap, (similarity, i))
        
        # Sort the heap to get the top similarities in reverse order
        test_data_item['similarity_list'] = [{'index': i, 'similarity': sim} for sim, i in sorted(heap, reverse=True)]
        test_data_item['leakage_ratio'] = numpy.mean([x['similarity'] for x in test_data_item['similarity_list']])

    leakage_ratio_mean = numpy.mean([item['leakage_ratio'] for item in test_data])
    print(f'TLI-mean@{top} : {leakage_ratio_mean}')
    leakage_ratio_max = numpy.max([item['leakage_ratio'] for item in test_data])
    print(f'TLI-max@{top} : {leakage_ratio_max}')

    # Remove grams field to save memory
    for item in train_data + test_data:
        del item['grams']

def filter_leaked_data(train_data, args):
    new_train_data = []
    leaked_data = []
    for item in train_data:
        if item['max_similarity'] >= args.threshold:
            leaked_data.append(item)
        else:
            new_train_data.append(item)
    return new_train_data, leaked_data


def analyis_data(train_data, test_data, args):
    leakage_info = []
    for item in test_data:
        test_text = extract_texts(item, args.key_test)
        leakage_info_item = {
                    'test_text': test_text,
                    'similarity_text_list': []}
        for i in range(args.top):
            train_index = item['similarity_list'][i]['index']
            train_item = train_data[train_index]
            leakage_info_item['similarity_text_list'].append((extract_texts(train_item, args.key_train), item['similarity_list'][i]['similarity']))
        leakage_info.append(leakage_info_item)
        leakage_info_item['max_similarity'] = leakage_info_item['similarity_text_list'][0][1]
    leakage_info = sorted(leakage_info, key=lambda x: x['max_similarity'], reverse=True)
    return leakage_info

def main():
    parser = argparse.ArgumentParser(description='Detect data leakage and overlap between training and test datasets')
    parser.add_argument('--train_data_path', type=str, help='Path to training data, which should be in JSONL format')
    parser.add_argument('--key_train', type=str, default=['messages'], nargs='+' , help='Key name of the instruction in the training data JSON, supports data stored in messages format and will automatically concatenate each round of user content')
    parser.add_argument('--test_data_path', type=str, help='Key name of the instruction in the test data JSON')
    parser.add_argument('--key_test', type=str, default=['prompt'], nargs='+', help='Which field of the test set to use for detection')
    parser.add_argument('--only_analysis', type=bool, default=False, help='Whether to perform filtering, if True, only calculate the TLI index between the two datasets without filtering')
    parser.add_argument('--clean_train_data_path', type=str, default=None, help='Path to save the filtered data')
    parser.add_argument('--leaked_data_path', type=str, default=None, help='Path to save the filtered out data')
    parser.add_argument('--leak_info_path', type=str, default=None, help='Store the top similar training data for each test data in the test set')
    parser.add_argument('--top', type=int, default=1, help='Count the top most similar data')
    parser.add_argument('--ngram_sizes', type=int, nargs='+', default=[4, 5], help='List of n-gram sizes to generate, default is 4, 5')
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold value, ranging from 0-1, considered similar if greater than this threshold and will be filtered out')
    args = parser.parse_args()

    # If no output file path is specified, handle it automatically according to the following rules
    tag = ''
    if 'mbpp' in args.test_data_path.lower():
        tag = 'mbpp'
    if 'humaneval' in args.test_data_path.lower() or 'human_eval' in args.test_data_path.lower():
        tag = 'humaneval'
    # If the path to save filtered data is not specified, handle it automatically
    if args.clean_train_data_path == None:
        args.clean_train_data_path = args.train_data_path.split('.json')[0]+f'_{args.threshold}_{tag}_clean.jsonl'    
    # If the path to save filtered out data is not specified, handle it automatically
    if args.leaked_data_path == None:
        args.leaked_data_path = args.train_data_path.split('.json')[0]+f'_{args.threshold}_{tag}_leaked.jsonl'    
    if args.leak_info_path == None:
        args.leak_info_path = args.train_data_path.split('.json')[0]+ f'_{args.threshold}_{tag}_leak_info.jsonl'


    # Read training data and test data
    train_data = read_jsonl(args.train_data_path)
    test_data = read_jsonl(args.test_data_path)


    # Calculate TLI before filtering
    calculate_similarity(train_data, test_data, args)

    

    if not args.only_analysis:
        # Count the top most similar training data for each test data in the test set
        leakage_info = analyis_data(train_data, test_data, args)

        # Start filtering
        new_train_data, leaked_data = filter_leaked_data(train_data, args)

        # Recalculate the TLI of the filtered data
        calculate_similarity(new_train_data, test_data, args)

        # Write the filtered data
        if args.clean_train_data_path:
            write_jsonl(args.clean_train_data_path, new_train_data)

        # Write the filtered out data
        if args.leaked_data_path:
            write_jsonl(args.leaked_data_path, leaked_data)

        # Write the information of each filtered out data and its most similar data in the test set
        if args.leak_info_path:
            write_jsonl(args.leak_info_path, leakage_info)

main()
