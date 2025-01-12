# Import the tkinter library for GUI development
import tkinter as tk
from tkinter import filedialog  # For file selection dialog
from tkinter import ttk         # For advanced GUI widgets like Combobox

# Import data manipulation libraries
import pandas as pd             # For loading and processing CSV data
import numpy as np              # For numerical computations

# Import TensorFlow for building and training the model
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization  # For text preprocessing
from tensorflow.keras import Model                     # For defining the model
from tensorflow.keras.optimizers import Adam           # For compiling the model

# Import pickle for saving and loading Python objects
import pickle

# Global variables to store mappings and the model
num_to_word_mapping = {}  # Dictionary to map English words to Spanish
model = None              # Placeholder for the trained model

# Function to save data (e.g., mappings or model) using pickle
def save_pickle(file_path, data):
    # Open the specified file in binary write mode and save the data
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Function to load data (e.g., mappings or model) using pickle
def load_pickle(file_path):
    # Open the specified file in binary read mode and return the data
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to load a dataset and train the translation model
def load_and_train(file_path):
    global num_to_word_mapping, model, english_sentences, spanish_sentences

    # Load the dataset from a CSV file
    data = pd.read_csv(file_path)
    # Extract English and Spanish sentences into separate lists
    english_sentences = data['english'].tolist()
    spanish_sentences = data['spanish'].tolist()

    # Create a dictionary mapping English sentences to Spanish sentences
    num_to_word_mapping = {english: spanish for english, spanish in zip(english_sentences, spanish_sentences)}
    # Populate the Combobox with English sentences
    word_combobox['values'] = english_sentences
    # Update the result label to indicate successful dataset loading
    result_label.config(text="Dataset loaded and mappings created successfully!")
    # Save the mappings using pickle
    save_pickle("mappings.pkl", num_to_word_mapping)

    # Text vectorization for preprocessing English and Spanish sentences
    english_vectorizer = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20)
    spanish_vectorizer = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20)
    # Fit the vectorizers to the respective sentences
    english_vectorizer.adapt(english_sentences)
    spanish_vectorizer.adapt(spanish_sentences)
    # Convert sentences into numerical sequences
    english_sequences = english_vectorizer(english_sentences)
    spanish_sequences = spanish_vectorizer(spanish_sentences)
    # Prepare the decoder input sequences by shifting and padding
    decoder_input_sequences = np.pad(spanish_sequences[:, :-1], ((0, 0), (1, 0)), mode='constant')

    # Build and compile the Transformer-based model
    model = build_model(vocab_size=1000, num_heads=4, key_dim=64, ff_dim=256, max_length=20)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Train the model on the dataset
    model.fit([english_sequences, decoder_input_sequences], np.expand_dims(spanish_sequences, -1), epochs=3)
    # Update the result label to indicate successful training
    result_label.config(text="Model trained successfully!")
    # Save the trained model
    model.save("translator_model.h5")

# Function to build a Transformer-based translation model (Encoder-Decoder architecture)
def build_model(vocab_size, num_heads, key_dim, ff_dim, max_length):
    # Define the input layers for the encoder and decoder
    encoder_input = tf.keras.Input(shape=(max_length,))
    decoder_input = tf.keras.Input(shape=(max_length,))
    # Embedding layers for encoder and decoder
    encoder_emb = tf.keras.layers.Embedding(vocab_size, key_dim)(encoder_input)
    decoder_emb = tf.keras.layers.Embedding(vocab_size, key_dim)(decoder_input)

    # Encoder: Multi-head self-attention followed by normalization
    encoder_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(encoder_emb, encoder_emb)
    encoder_output = tf.keras.layers.LayerNormalization()(encoder_attn)

    # Decoder: Attention over encoder output and normalization
    decoder_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(decoder_emb, encoder_output)
    decoder_output = tf.keras.layers.LayerNormalization()(decoder_attn)
    # Feedforward network for processing decoder output
    ffn_output = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(vocab_size)
    ])(decoder_output)

    # Output layer with softmax activation for probability distribution over vocabulary
    output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')(ffn_output)

    # Define the model with encoder and decoder inputs and output layer
    model = Model(inputs=[encoder_input, decoder_input], outputs=output_layer)
    return model

# Function to translate the selected English sentence to Spanish
def translate_number():
    global num_to_word_mapping

    # Get the selected English sentence from the Combobox
    selected_english_word = word_combobox.get().strip()
    # Check if a sentence is selected and mappings are available
    if not selected_english_word or not num_to_word_mapping:
        result_label.config(text="Please load a dataset and select an English word to translate!")
        return

    # Get the Spanish translation from the mappings dictionary
    translated_word = num_to_word_mapping.get(selected_english_word, "Translation not found!")
    # Display the translated word in the result label
    result_label.config(text=f"Spanish Translation: {translated_word}")

# Function to load mappings from a saved pickle file
def load_mappings():
    global num_to_word_mapping

    try:
        # Load the mappings dictionary from the pickle file
        num_to_word_mapping = load_pickle("mappings.pkl")
        # Populate the Combobox with English sentences from the mappings
        word_combobox['values'] = list(num_to_word_mapping.keys())
        result_label.config(text="Mappings loaded successfully!")
    except FileNotFoundError:
        result_label.config(text="Mappings file not found. Please load a dataset first!")

# Function to load a pre-trained model
def load_model():
    global model

    try:
        # Load the trained model from the saved file
        model = tf.keras.models.load_model("translator_model.h5")
        result_label.config(text="Model loaded successfully!")
    except FileNotFoundError:
        result_label.config(text="Model file not found. Please train a model first!")

# Function to load a dataset file using a file dialog
def load_file():
    # Open a file selection dialog for CSV files
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        # Call the function to load and train the dataset
        load_and_train(file_path)

# GUI Setup
window = tk.Tk()
window.title("English to Spanish Translator")  # Set the window title
window.geometry("500x400")                     # Set the window dimensions

# Create and place labels, buttons, and widgets for the GUI
menu_label = tk.Label(window, text="Select English Sentence:")
menu_label.pack()

word_combobox = ttk.Combobox(window, width=40)
word_combobox.pack()
translate_button = tk.Button(window, text="Translate", command=translate_number)
translate_button.pack()
result_label = tk.Label(window, text="Spanish Translation will appear here.", wraplength=400, bg="lightgray", height=5)
result_label.pack(fill="both", expand=True)
dataset_label = tk.Label(window, text="Load Dataset:")
dataset_label.pack()
dataset_button = tk.Button(window, text="Load Dataset", command=load_file)
dataset_button.pack()
load_mappings_button = tk.Button(window, text="Load Mappings", command=load_mappings)
load_mappings_button.pack()
load_model_button = tk.Button(window, text="Load Model", command=load_model)
load_model_button.pack()

# Start the main event loop for the GUI
window.mainloop()
