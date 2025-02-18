import json
from typing import Union
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer


DATASET_PATH = '../data/ShareGPT_V3_unfiltered_cleaned_split.json'
MODEL_PATH = '../models/llama-3.1-8b'


def get_tokenizer(
    pretrained_model_name_or_path: str
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)


def sample_sharegpt_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    count = 0
    sum_input_length = 0
    sum_output_length = 0
    for index_dataset, data in enumerate(dataset):
        print(f'Pending {len(dataset) - index_dataset} conversations')

        # filter out the conversations with less than 2 turns.
        if len(data['conversations']) < 2:
            continue

        # only keep the first two turns of each conversation.
        # do nothing

        index = 0
        while index < len(data['conversations']):
            if index + 1 >= len(data['conversations']):
                break

            '''if data['conversations'][index]['from'] != 'human':
                break
            if data['conversations'][index + 1]['from'] != 'gpt':
                break'''

            prompt = data['conversations'][index]['value']
            completion = data['conversations'][index + 1]['value']

            prompt_token_ids = tokenizer(prompt).input_ids
            completion_token_ids = tokenizer(completion).input_ids

            prompt_len = len(prompt_token_ids)
            output_len = len(completion_token_ids)

            sum_input_length += prompt_len
            sum_output_length += output_len

            count += 1
            index += 2

    mean_input_length = sum_input_length / count
    mean_output_length = sum_output_length / count

    print(f'Number of valid conversations: {count}')
    print(f'Mean input length: {mean_input_length}')
    print(f'Mean output length: {mean_output_length}')


def main():
    tokenizer = get_tokenizer(MODEL_PATH)
    sample_sharegpt_requests(DATASET_PATH, tokenizer)


if __name__ == "__main__":
    main()
