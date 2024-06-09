import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for dialog in data:
            if dialog['input'].strip() == '' or dialog['output'].strip() == '':
                continue
            input_ids = tokenizer(dialog['input'], return_tensors='pt', truncation=True, padding='max_length',
                                  max_length=max_length).input_ids
            output_ids = tokenizer(dialog['output'], return_tensors='pt', truncation=True, padding='max_length',
                                   max_length=max_length).input_ids
            if input_ids.size(-1) == max_length and output_ids.size(-1) == max_length:
                self.input_ids.append(input_ids)
                self.attention_masks.append(torch.ones_like(input_ids))
                self.labels.append(output_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].squeeze(),
            'attention_mask': self.attention_masks[idx].squeeze(),
            'labels': self.labels[idx].squeeze()
        }


def train_model(data_file, model_name='ytu-ce-cosmos/Turkish-Llama-8b-v0.1', output_dir='./results'):
    print("Loading training data...")
    with open(data_file, 'r', encoding='utf-8') as file:
        training_data = json.load(file)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Creating dataset...")
    dataset = ChatDataset(training_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced for debugging
        per_device_train_batch_size=1,  # Reduced for debugging
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")

    # Save the model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved.")
