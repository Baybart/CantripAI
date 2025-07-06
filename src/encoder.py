from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

class Encoder:
    # batch iterator should be defined in dataset.py
    # you can add as need arises
    def __init__(self, batch_iterator, vocab_size, special_tokens):
        self.batch_iterator = batch_iterator

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        self.trainer = trainers.BpeTrainer(
            vocab_size = 20000, 
            min_frequency = 2, 
            show_progress = True, 
            special_tokens = special_tokens
            )

    def train_encoder(self, savepath):
        self.tokenizer.train_from_iterator(self.batch_iterator, trainer=self.trainer)
        self.tokenizer.save(savepath)