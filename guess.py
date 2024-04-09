import torch
from transformers import BertTokenizer, BertForMaskedLM
def suggest_next_letter(displayed_word, guessed_letters, model, tokenizer):
    """
    This function takes in the current state of the game and returns the next letter to be guessed.
    This is based on the BERT model provided.

    Args:
    - displayed_word (str): The word being guessed, with underscores for unguessed letters.
    - guessed_letters (list): A list of the letters that have been guessed so far.
    - model (BertForMaskedLM): The BERT model for prediction.
    - tokenizer (BertTokenizer): The BERT tokenizer.

    Returns:
    - predicted_letter (str): The next letter to be guessed.
    """
    sorted_letters=['e', 'i', 'a', 'n', 'r', 'o', 's', 't', 'l', 'c', 'u', 'd', 'p', 'm', 'h', 'g', 'y', 'b', 'f', 'v', 'k', 'w', 'z', 'x', 'q', 'j']

    # Tokenize the displayed_word
    max_length = 29  # Maximum length for displayed word
    tokenized_word = tokenizer(displayed_word, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    input_ids = tokenized_word.input_ids
    attention_mask = tokenized_word.attention_mask

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get predicted token index
    predicted_index = torch.argmax(outputs.logits[0, -1]).item()

    # Convert token index to letter
    predicted_letter = tokenizer.convert_ids_to_tokens([predicted_index])[0] 

    
    # Ensure that the predicted letter is not already guessed
    while predicted_letter is None or predicted_letter[0] in guessed_letters or not predicted_letter.islower():
        for letter in sorted_letters:
            if letter not in guessed_letters:
                predicted_letter =letter
                break

    return predicted_letter[0]

def play_move(displayed_word, guessed_letters):
    """
    If you want to play the game, you can use this function to play the game.
    
    Args:
    - displayed_word (str): The word being guessed, with underscores for unguessed letters.
    - guessed_letters (list): A list of the letters that have been guessed so far.

    Returns:
    - guess (str): The guessed letter.
    """
    guess = input("Enter the letter: ")
    return guess

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load the saved model state dictionary
model.load_state_dict(torch.load('modelKaggle2.pth', map_location=torch.device('cpu')))

if __name__ == "__main__":
    # You can call suggest_next_letter and play_move functions here for testing
    pass
