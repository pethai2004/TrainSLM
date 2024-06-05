
import os 
import shutil
import json 
import regex as re
import unicodedata
from datasets import load_dataset, Dataset

import transformers as tfm 


def flatten_dir_structure(directory: str):
    """Flatten the directory structure and remove subdirectories, and `.DS_Store` files"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
            else:
                shutil.move(os.path.join(root, file), os.path.join(directory, file))
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            shutil.rmtree(os.path.join(directory, item))
            
def recursive_func(inputs, func):
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs[key] = recursive_func(value, func)
        return inputs
    elif isinstance(inputs, (list, tuple)):
        return [
            recursive_func(value, func) for value in inputs
        ]
    elif isinstance(inputs, str):
        return func(inputs)
    else:
        return inputs
    
def regex_preprocessor(inputs):

    def _regex_preprocess(txt):
        txt = re.sub(r'<[^>]*>', "[HTML]", txt) # Remove HTML tags

        # Replace emojis and other special characters with an empty string (removal)
        txt = re.sub(r'[\U00010000-\U0010ffff]', ' ', txt)  # Replace emojis
        txt = ''.join([c if unicodedata.category(c) != 'So' else ' ' for c in txt])

        txt = re.sub(r'[^a-zA-Z0-9\s,.!?\'\"-]', ' ', txt) # Replace special characters with a space

        txt = re.sub(r'\b\d{10}\b', '[Phone Number]', txt)  # Replace 10-digit phone numbers
        txt = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[Phone Number]', txt)  # Replace phone numbers in format xxx-xxx-xxxx
        txt = re.sub(r'\+\d{1,3}\s?\d{1,14}', '[Phone Number]', txt)  # Replace international format phone numbers

        txt = re.sub(r'\S+@\S+', '[Email]', txt) # Replace emails with [Email]
        txt = re.sub(r'http\S+|www\S+', '[URL]', txt) # Replace URLs with [URL]
        txt = re.sub(r"b'(.*?)'", ' ', txt)  # Handles byte objects formatted as strings like "b'example'"
        # remove \n, \t, \r, \x0c, \x0b, and any \x00-\x1f characters
        txt = re.sub(r'[\t\r\x0c\x0b\x00-\x1f]', ' ', txt)
        # replace \n with a space
        txt = re.sub(r'\n', ' ', txt) # deal separately with \n
        txt = txt.strip()
        # replace numnber with [Number]
        txt = re.sub(r'\b\d+\b', '[Number]', txt)
        txt = re.sub(r'\s+', ' ', txt)  # Replace multiple spaces with a single space

        txt = txt.lower() # maybe make this optional
        
        return txt
    
    return recursive_func(inputs, _regex_preprocess)