import argparse
import os
from prepare_data import prepare_training_data
from train_model import train_model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model on your dataset.")
    parser.add_argument('--whatsapp_dir', type=str, required=True, help="Directory containing WhatsApp chat files.")
    parser.add_argument('--facebook_dir', type=str, required=True, help="Directory containing Facebook chat files.")
    parser.add_argument('--main_user', type=str, required=True, help="User name")
    parser.add_argument('--intermediate_file', type=str, default='data/intermediate_data.json', help="Path to the intermediate combined data JSON file.")
    parser.add_argument('--data_file', type=str, default='data/combined_training_data.json', help="Path to the final training data JSON file.")
    parser.add_argument('--model_name', type=str, default='ytu-ce-cosmos/Turkish-Llama-8b-v0.1', help="Model name or path.")
    parser.add_argument('--output_dir', type=str, default='./results', help="Directory to save the fine-tuned model.")

    args = parser.parse_args()

    # Ensure directories exist
    if not os.path.isdir(args.whatsapp_dir):
        print(f"WhatsApp directory {args.whatsapp_dir} does not exist.")
        return

    if not os.path.isdir(args.facebook_dir):
        print(f"Facebook directory {args.facebook_dir} does not exist.")
        return

    # Check if the data file already exists
    if not os.path.exists(args.data_file):
        print(f"{args.data_file} not found. Preparing training data...")
        prepare_training_data(args.whatsapp_dir, args.facebook_dir, args.intermediate_file, args.data_file, args.main_user)
    else:
        print(f"{args.data_file} already exists. Skipping data preparation.")

    # Train the model
    train_model(args.data_file, args.model_name, args.output_dir)

if __name__ == '__main__':
    main()
