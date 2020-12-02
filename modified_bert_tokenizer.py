from  transformers import BertTokenizer


class ModifiedBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
    
    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in text.split():
                split_tokens.append(token)
            #for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # If the token is part of the never_split set
                #if token in self.basic_tokenizer.never_split:
                #    split_tokens.append(token)
                #else:
                #    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens