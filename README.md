Hangman Game Predictor Model Report

Abstract:
Utilized BERT (Bidirectional Encoder Representations from Transformers) for predicting masked letters in the Hangman game. Objective was to effectively predict hidden letters.

Modeling Approach:
Data Preprocessing: Extracted individual words from training files and tokenized using BERT tokenizer.
Model Architecture: Utilized BertForMaskedLM pre-trained model fine-tuned for masked language modeling.
Training: Trained model on preprocessed data using AdamW optimizer with a learning rate of 5e-5 for 11 epochs.
Evaluation: Assessed model performance based on accurate prediction of masked letters using loss metrics and qualitative analysis.

Model Architecture:
BERT model comprised multiple transformer layers for effective contextual information capture.

Data Preparation and Preprocessing:
Tokenization: Employed BERT tokenizer to tokenize words ensuring each letter was tokenized separately with padding for uniform input lengths.
Masking: Randomly masked subset of tokens simulating Hangman game scenario.

Predictions:
Recursive Approach: Iteratively predicted masked letters in words based on surrounding context, updating input sequence for subsequent predictions.
