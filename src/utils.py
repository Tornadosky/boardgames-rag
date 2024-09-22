# src/utils.py
import os

def load_rules(data_folder):
    """Load board game rules from text files."""
    rule_files = ["monopoly.txt", "battleship.txt", "ticket_to_ride.txt", "codenames.txt", "kittens.txt"]
    rules = {}
    for file_name in rule_files:
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, "r") as file:
            rules[file_name.split('.')[0]] = file.read()
    return rules

def text_formatter(text):
    """Format text by removing excess whitespace and line breaks."""
    import re
    cleaned_text = text.replace("\n", " ").replace("\t", " ").strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text
