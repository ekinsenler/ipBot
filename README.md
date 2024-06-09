# LLaMA Chatbot Fine-Tuning and Deployment

This project fine-tunes a LLaMA (Large Language Model) chatbot on custom chat data from WhatsApp and Facebook Messenger. It processes the chat data, trains the model, and provides a web interface for interacting with the fine-tuned chatbot.

## Table of Contents
- [Directory Structure](#directory-structure)
- ipBot/
│
├── data/
│   ├── messenger/
│   │   ├── message_1.html
│   │   ├── message_2.html
│   │   └── ...
│   ├── whatsapp/
│   │   ├── whatsapp-1.txt
│   │   ├── whatsapp-2.txt
│   │   └── ...
│   └── intermediate_data.json
│   └── combined_training_data.json
│
├── results/
│   └── (fine-tuned model and tokenizer)
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── prepare_data.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
- [Data Preparation](#data-preparation)
- Place your WhatsApp and Facebook Messenger chat files in the data/whatsapp and data/messenger directories, respectively.
- This script will combine the chat data into a single JSON file and create training input-output pairs.
- [Training the Model](#training-the-model)
- Run the training script:
```bash
python main.py --whatsapp_dir data/whatsapp --facebook_dir data/messenger --intermediate_file data/intermediate_data.json --data_file data/combined_training_data.json --model_name ytu-ce-cosmos/Turkish-Llama-8b-v0.1 --output_dir ./results
```
- [Deployment](#deployment)
- Run the deployment script:
```bash
python app.py --model_dir ./results
```
- [License](#license)
