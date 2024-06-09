import os
import json
import re
from bs4 import BeautifulSoup

def parse_facebook_messenger_html(file_path):
    messages = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            for message in soup.find_all('div', class_='_a6-g'):
                user = message.find('div', class_='_2ph_ _a6-h _a6-i').get_text()
                content = message.find('div', class_='_2ph_ _a6-p').get_text()
                timestamp = message.find('div', class_='_3-94 _a6-o').get_text()
                messages.append({"timestamp": timestamp, "user": user, "message": content})
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return messages

def parse_facebook_messenger_directory(directory_path):
    all_messages = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.html'):
            file_path = os.path.join(directory_path, filename)
            messages = parse_facebook_messenger_html(file_path)
            all_messages.extend(messages)

    return all_messages

def clean_whatsapp_chat(file_path):
    pattern = re.compile(r'\[(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2})\] (.+?): (.*)')
    messages = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = pattern.match(line)
                if match:
                    timestamp, user, message = match.groups()
                    messages.append({"timestamp": timestamp, "user": user, "message": message})
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return messages

def prepare_training_data(whatsapp_dir, facebook_dir, intermediate_file, output_file, main_user):
    whatsapp_messages = []
    facebook_messages = []

    # Read all WhatsApp chat files
    for filename in os.listdir(whatsapp_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(whatsapp_dir, filename)
            whatsapp_messages.extend(clean_whatsapp_chat(file_path))

    # Read all Facebook chat files
    for filename in os.listdir(facebook_dir):
        if filename.endswith('.html'):
            file_path = os.path.join(facebook_dir, filename)
            facebook_messages.extend(parse_facebook_messenger_html(file_path))

    # Combine messages
    all_messages = whatsapp_messages + facebook_messages

    # Save the combined messages to an intermediate JSON file
    try:
        with open(intermediate_file, 'w', encoding='utf-8') as file:
            json.dump(all_messages, file, ensure_ascii=False, indent=4)
        print(f"Intermediate data saved to {intermediate_file}")
    except Exception as e:
        print(f"Error saving intermediate data: {e}")

    # Create training data
    create_training_data(all_messages, output_file, main_user)

def create_training_data(messages, output_file, main_user):
    conversations = []
    current_input = ""
    current_output = ""
    input_active = True

    for i in range(len(messages)):
        message_user = messages[i]["user"]
        message_text = messages[i]["message"]

        if main_user in message_user:
            if not input_active:
                # We switch back to main user, save the current conversation
                if current_input and current_output:
                    conversations.append({"input": current_input.strip(), "output": current_output.strip()})
                current_input = message_text
                current_output = ""
                input_active = True
            else:
                current_input += " " + message_text
        else:
            if input_active:
                # We switch away from main user, start collecting output
                current_output = message_text
                input_active = False
            else:
                current_output += " " + message_text

    # Save any remaining conversation
    if current_input and current_output:
        conversations.append({"input": current_input.strip(), "output": current_output.strip()})

    # Save the training data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(conversations, file, ensure_ascii=False, indent=4)

    print(f"Training data saved to {output_file}")
