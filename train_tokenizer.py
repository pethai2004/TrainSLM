
import os 
import json

from tokenizers import Tokenizer, models, trainers, decoders
from tokenizers.pre_tokenizers import Metaspace, Punctuation
from tokenizers.normalizers import NFKC, Lowercase
from tokenizers.normalizers import Sequence as norm_sequence
from tokenizers.pre_tokenizers import Sequence as pre_sequence

def _get_trained_tokenizer(
        path_to_data_txt="podcast_text.txt",
        vocab_size: int=12_000,
        min_frequency: int=10,
        save_path: str="DefaultTokenizer",
        special_tokens: list=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    ):
    if not isinstance(path_to_data_txt, list):
        path_to_data_txt = [path_to_data_txt]
        
    if os.path.exists(save_path):
        raise ValueError(f"Save path already exists for {save_path}")
    
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = norm_sequence([NFKC(), Lowercase()])
    tok.pre_tokenizer = pre_sequence([Metaspace(), Punctuation()])
    tok.decoder = decoders.Metaspace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        max_token_length=20
    )
    
    tok.train(path_to_data_txt, trainer)
    tok.save(save_path)

if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/Users/owan/Documents/BloggingOFman/SLM_/CombinedDataset.txt")
    parser.add_argument("--vocab_size", type=int, default=20000)
    path = parser.parse_args().path
    vocab_size = parser.parse_args().vocab_size
    
    _get_trained_tokenizer(
        path_to_data_txt=path,
        vocab_size=vocab_size,
    )