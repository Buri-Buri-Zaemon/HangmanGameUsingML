from guess import suggest_next_letter, play_move
from transformers import BertTokenizer, BertForMaskedLM


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


def suggest_next_letter_sol(displayed_word, guessed_letters):
    """_summary_

    This function takes in the current state of the game and returns the next letter to be guessed.
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    Use python hangman.py to check your implementation.
    """
    ################################################
    ################################################
    ################################################
    ''' Letters sorted by occurrence frequency (most to least): ['e', 'i', 'a'
    , 'n', 'r', 'o', 's', 't', 'l', 'c', 'u', 'd', 'p', 'm', 'h', 'g', 'y',
    'b','f', 'v', 'k', 'w', 'z', 'x', 'q', 'j'] found from our file
    '''
    sorted_letters=['e', 'i', 'a', 'n', 'r', 'o', 's', 't', 'l', 'c', 'u', 'd', 'p', 'm', 'h', 'g', 'y', 'b', 'f', 'v', 'k', 'w', 'z', 'x', 'q', 'j']

    # Calculate the number of dashes in the displayed word
    num_dashes = displayed_word.count('_')
    
    # Calculate the percentage of dashes in the displayed word
    dash_percentage = num_dashes / len(displayed_word)
    
    # If the dash percentage exceeds the threshold, guess the most frequent letter
    if dash_percentage <= 0.2 :
        while True:
            next_letter = suggest_next_letter(displayed_word, guessed_letters, model, tokenizer)
            if next_letter not in guessed_letters:
                return next_letter
        
        # Guess the first letter of the predicted word
        if next_letter:
            return
    else:

        # Iterate through sorted letters and return the first one not guessed
        for letter in sorted_letters:
            if letter not in guessed_letters:
                return letter
                # Use ML model for word prediction

    ################################################
    ################################################
    ################################################
    raise NotImplementedError