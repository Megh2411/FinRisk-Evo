import os
import sys
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

MAGNITUDE_WORDS = {
    "increase", "increased", "increasing", "surge", "surged",
    "decrease", "decreased", "decreasing", "plunge", "plunged",
    "drop", "dropped", "dip", "dipped", "rise", "rose", "rising",
    "fall", "fell", "falling", "grow", "grew", "growth", "decline", 
    "declined", "skyrocket", "plummet", "plummeted", "jump", "jumped"
}

def load_and_prepare_data(file_path: str):
    """Loads the extracted TNCRM text file into a HuggingFace Dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return Dataset.from_dict({"text": lines})

def train_tncrm_finbert(input_file: str, output_dir: str):
    print("Initializing Precision Engine (Module B)...")
    
    # 1. Load Base FinBERT Model and Tokenizer
    model_checkpoint = "ProsusAI/finbert"
    print(f"Downloading/Loading base model: {model_checkpoint}")
    
    # We suppress the warnings in production, but you know the new head is initializing
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # 2. Load the TNCRM Dataset
    dataset = load_and_prepare_data(input_file)
    print(f"Loaded dataset with {len(dataset)} sentences.")

    # 3. Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_special_tokens_mask=True
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

class TNCRMDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, magnitude_words, mlm_probability=0.15, target_probability=0.75, baseline_probability=0.10, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, **kwargs)
        self.target_probability = target_probability
        self.baseline_probability = baseline_probability
        
        # Identify targeted token IDs in vocabulary
        self.targeted_token_ids = set()
        vocab = tokenizer.get_vocab()
        
        mag_set = {w.lower() for w in magnitude_words}
        for token_str, token_id in vocab.items():
            # Check if token is numerical or contains numbers
            has_digits = any(char.isdigit() for char in token_str)
            # Remove ## for subword matching
            clean_token = token_str.replace("##", "").lower()
            is_magnitude = clean_token in mag_set
            
            if has_digits or is_magnitude:
                self.targeted_token_ids.add(token_id)
                
        print(f"TNCRM Collator initialized: Identified {len(self.targeted_token_ids)} targeted tokens out of {len(vocab)} vocab tokens.")

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Custom masking logic that forces high probability masking on numerical/magnitude tokens.
        """
        import torch

        labels = inputs.clone()
        
        # Initialize probability matrix with baseline probability
        probability_matrix = torch.full(labels.shape, self.baseline_probability, device=inputs.device)
        
        # Construct mask of special tokens
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Set special tokens probability to 0.0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Find where targeted tokens occur in input
        targeted_mask = torch.zeros(labels.shape, dtype=torch.bool, device=inputs.device)
        for token_id in self.targeted_token_ids:
            targeted_mask |= (labels == token_id)
            
        # Apply the high target masking probability to those tokens
        probability_matrix = torch.where(targeted_mask, self.target_probability, probability_matrix)
        
        # Mask inputs where random Bernoulli trial succeeds
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # We only compute loss on masked tokens (unmasked labels are set to -100)
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random words
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The remaining 10% of the time, we keep the masked input tokens unchanged
        return inputs, labels

def train_tncrm_finbert(input_file: str, output_dir: str):
    print("Initializing Precision Engine (Module B)...")
    
    # 1. Load Base FinBERT Model and Tokenizer
    model_checkpoint = "ProsusAI/finbert"
    print(f"Downloading/Loading base model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # 2. Load the TNCRM Dataset
    dataset = load_and_prepare_data(input_file)
    print(f"Loaded dataset with {len(dataset)} sentences.")

    # 3. Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_special_tokens_mask=True
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Custom TNCRM Collator (Ensuring high sensitivity to numbers and magnitude markers)
    data_collator = TNCRMDataCollator(
        tokenizer=tokenizer,
        magnitude_words=MAGNITUDE_WORDS,
        target_probability=0.75,
        baseline_probability=0.10
    )

    # 5. Define Training Arguments
    # Removed overwrite_output_dir to fix the TypeError
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        learning_rate=2e-5,
        remove_unused_columns=False, # Prevents issues with custom datasets
        fp16=torch.cuda.is_available(), 
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # 7. Execute Post-Training
    print("\nStarting TNCRM Post-Training. This will update FinBERT's attention weights...")
    trainer.train()

    # 8. Save the Numerically-Enhanced Model
    print(f"\nTraining complete. Saving Numerically-Enhanced FinBERT to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "03_processed")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "src", "models", "finbert_tncrm_weights")
    
    input_file = os.path.join(PROCESSED_DIR, "Mahindra_FY24_tncrm_corpus.txt")
    
    if os.path.exists(input_file):
        train_tncrm_finbert(input_file, MODEL_OUTPUT_DIR)
    else:
        print(f"Cannot find TNCRM corpus at: {input_file}")