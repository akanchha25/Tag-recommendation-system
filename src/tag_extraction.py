# Import statements
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import itertools
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("efederici/text2tags")
tokenizer = AutoTokenizer.from_pretrained("efederici/text2tags")

article = '''
Life is full of ups and downs. It's a journey with good and bad moments. We all experience happiness, sadness, and challenges. It's about learning, growing, and finding joy in everyday things. Life is about family, friends, and the people we meet along the way. We strive to make the most of our time, create memories, and be kind to one another. It's a precious gift that we should cherish and make the most of every day.
'''

def words(text):
    input_str = text
    output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
    return output_str.split()

def is_subset(text1, text2):
    return all(tag in words(text1.lower()) for tag in text2.split())

def cleaning(text, tags):
    return [tag for tag in tags if is_subset(text, tag)]

def get_texts(text, max_len):
    texts = list(filter(lambda x: x != '', text.split('\n\n')))
    lengths = [len(tokenizer.encode(paragraph)) for paragraph in texts]
    output = []
    for i, par in enumerate(texts):
        index = len(output)
        if index > 0 and lengths[i] + len(tokenizer.encode(output[index-1])) <= max_len:
            output[index-1] = "".join(output[index-1] + par)
        else:
            output.append(par)
    return output

def get_tags(text, generate_kwargs):
    input_text = 'summarize: ' + text.strip().replace('\n', ' ')
    tokenized_text = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        tags_ids = model.generate(tokenized_text, **generate_kwargs)

    output = []
    for tags in tags_ids:
        cleaned = cleaning(
            text,
            list(set(tokenizer.decode(tags, skip_special_tokens=True).split(', ')))
        )
        output.append(cleaned)

    return list(set(itertools.chain(*output)))

def tag(text, max_len, generate_kwargs):
    texts = get_texts(text, max_len)
    all_tags = [get_tags(text, generate_kwargs) for text in texts]
    flatten_tags = itertools.chain(*all_tags)
    return list(set(flatten_tags))

params = {
    "min_length": 0,
    "max_length": 30,
    "no_repeat_ngram_size": 2,
    "num_beams": 4,
    "early_stopping": True,
    "num_return_sequences": 4,
}
tags = tag(article, 512, params)
print(tags)
