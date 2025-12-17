"""
preprocessing.py

Handles the ETL (Extract, Transform, Load) pipeline for the CRD3 dataset.
It reads aligned JSON transcripts, tokenizes them using a sliding window approach,
masks non-target speakers (User Masking), and saves a Hugging Face Dataset to disk.
"""

import json
import glob
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import os

from utils import CRD3_DATA_DIR , PROCESSED_DATA_DIR

# --- CONFIGURATION ---
DATA_PATH = CRD3_DATA_DIR / "aligned data/c=2/"
MODEL_NAME = "gpt2"
GM_NAME = "MATT"

# Designate as None if you don't wish to save
SAVE_PATH = PROCESSED_DATA_DIR

TARGET_EPISODES = ["C1E001", "C1E002", "C1E003", "C1E004", "C1E005"]

MAX_SEQ_LEN = 1024

class CRD3Utterance:
    """
    Represents a single turn in the dialogue. 
    Handles text formatting and speaker-specific logic.
    """
    def __init__(self, speaker: str, text: list[str]):
        self.speaker = speaker
        self.text_content = " ".join(text) 
        self.full_formatted_text = f"{self.speaker}: {self.text_content}\n"
        
        self.token_ids = []
        self.mask_map = [] 

    def process(self, tokenizer, target_speaker=GM_NAME):
        """
        Tokenizes the text and generates the training mask.
        
        Args:
            tokenizer: The model-specific tokenizer.
            target_speaker: The speaker name we want to clone (mask=1). 
                            All others are ignored (mask=0).
        Returns:
            int: The length of the tokenized sequence.
        """
        self.token_ids = tokenizer.encode(self.full_formatted_text, add_special_tokens=False)
        
        # Create Mask: 1 for Target (GM), 0 for Others (Players)
        is_target = target_speaker in self.speaker
        
        if is_target:
            self.mask_map = [1] * len(self.token_ids)
        else:
            self.mask_map = [0] * len(self.token_ids)
            
        return len(self.token_ids)
    

class TokenSteamer:
    """
    A sliding-window buffer. Accumulates tokens from multiple utterances 
    and yields fixed-size chunks for training.
    """
    def __init__(self, tokenizer, max_seq_len = MAX_SEQ_LEN, stride = None):
        self.max_seq_len = max_seq_len
        self.stride = stride if stride else max_seq_len
        self.tokenizer = tokenizer

        # FIFO Buffers
        self.id_buffer = []
        self.mask_buffer = []

    def add(self, utterance):
        """Push a processed utterance into the buffer."""
        self.id_buffer.extend(utterance.token_ids)
        self.mask_buffer.extend(utterance.mask_map)

    def has_chunk(self):
        """Check if buffer has enough data for a full context window."""
        return len(self.id_buffer) >= self.max_seq_len
    
    def pop_chunk(self):
        """
        Extracts one max_seq_len chunk, applies -100 masking, and slides window.
        """
        # 1. Slice the window
        chunk_ids = self.id_buffer[:self.max_seq_len]
        chunk_mask = self.mask_buffer[:self.max_seq_len]

        # 2. Apply Masking Logic (-100 for ignored tokens)
        labels = list(chunk_ids)
        for i, train_check in enumerate(chunk_mask):
            if train_check == 0:
                labels[i] = -100

        # 3. Slide the window forward (consume tokens)
        self.id_buffer = self.id_buffer[self.stride:]
        self.mask_buffer = self.mask_buffer[self.stride:]

        return {
            "input_ids": chunk_ids,
            "labels": labels,
            "attention_mask": [1] * len(chunk_ids)
        }

    def flush(self):
        """
        Pads and returns the remaining data in the buffer (if any).
        """
        real_data_len = len(self.id_buffer)
        if real_data_len == 0: return None
        padding_len = self.max_seq_len-real_data_len

        # Determine Pad Token (use EOS if PAD is missing)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Prepare Padding Arrays
        padding_ids = [pad_id]*padding_len
        padding_labels = [-100]*padding_len

        # Prepare Data Arrays
        chunk_ids = self.id_buffer
        chunk_mask = self.mask_buffer
        labels = list(chunk_ids)

        # Apply Masking to real data
        for i, train_check in enumerate(chunk_mask):
            if train_check == 0:
                labels[i] = -100
        
        # Combine Real + Padding
        chunk_ids.extend(padding_ids)
        labels.extend(padding_labels)
        
        return {
            "input_ids": chunk_ids,
            "labels": labels,
            # Mask: 1 for Real Data, 0 for Padding
            "attention_mask": [1]*real_data_len +[0]*padding_len
        }
    

def CRD3_generator(paths, tokenizer, max_seq_len = MAX_SEQ_LEN):
    """
    The main generator function required by Dataset.from_generator.
    Reads files -> Processes Utterances -> Streams Chunks.
    """
    streamer = TokenSteamer(tokenizer, max_seq_len)

    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Iterate through the episode structure 
        for chunk in data:
            for turn in chunk['TURNS']:
                # 1. Create Object
                obj = CRD3Utterance(
                    speaker=turn['NAMES'][0] if turn['NAMES'] else "UNKNOWN",
                    text=turn['UTTERANCES']
                )
                
                # 2. Process & Add to Stream
                obj.process(tokenizer, target_speaker="MATT")
                streamer.add(obj)

                # 3. Yield full chunks immediately
                while streamer.has_chunk():
                    yield streamer.pop_chunk()

    # 4. Flush leftovers
    last_chunk = streamer.flush()
    if last_chunk:
        yield last_chunk


def construct_dataset():
    print(f"Initializing Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- FILTER LOGIC ---
    all_files = glob.glob(str(DATA_PATH / "*.json"))
    
    filtered_files = [f for f in all_files if any(ep in os.path.basename(f) for ep in TARGET_EPISODES)]
    print(DATA_PATH)
    print(f"[DEBUG] glob.glob found {len(all_files)} files.")
    filtered_files.sort()

    if not filtered_files:
        raise FileNotFoundError(f"No matching files found for {TARGET_EPISODES} at {DATA_PATH}")
    
    print(f"Selected {len(filtered_files)} episodes: {[os.path.basename(f) for f in filtered_files]}")

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int64")),
    })

    dataset = Dataset.from_generator(
        generator=CRD3_generator,
        features=features,
        gen_kwargs={
            "paths": filtered_files,
            "tokenizer": tokenizer,
            "max_seq_len": MAX_SEQ_LEN
        })
    
    if SAVE_PATH:
        print(f"Saving dataset to disk: {SAVE_PATH}...")
        dataset.save_to_disk(str(SAVE_PATH))
        print("Save complete.")

    return dataset

if __name__ == "__main__":
    construct_dataset()