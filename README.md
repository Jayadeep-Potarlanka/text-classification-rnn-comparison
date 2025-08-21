# Text Classification with RNN Architectures

This project develops and evaluates text classification models for two distinct datasets using four different recurrent neural network architectures. The goal is to predict the topic or domain of a given text sample.

## Objective

The primary objective is to build, train, and evaluate classification models for two separate datasets. The project implements four different non-transformer-based algorithms (SimpleRNN, LSTM, GRU, and Bi-LSTM) and compares their performance to analyze their respective strengths and weaknesses in text classification tasks.

## Datasets

### 1. Dataset 1: Research Article Abstracts
*   **Content:** This dataset contains the titles and abstracts of research articles.
*   **Task:** The goal is to classify each article into one of six topics: Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, or Quantitative Finance.
*   **Preprocessing:** The abstracts were tokenized, and pre-trained GloVe word embeddings were used to create feature vectors.

### 2. Dataset 2: A2D2 - Multi-Domain Articles
*   **Content:** This dataset contains the titles and content of articles from various domains.
*   **Task:** The goal is to classify each article into one of five domains: Entertainment, Healthcare, Sports, Technology, or Tourism.
*   **Preprocessing:** The text was preprocessed using spaCy for lemmatization and stop-word removal. Class weights were applied during training to handle the imbalanced nature of the dataset.

## Algorithms Implemented

The following four algorithms were implemented and evaluated for both datasets:
1.  **Simple Recurrent Neural Network (RNN):** A basic recurrent network.
2.  **Long Short-Term Memory (LSTM):** An advanced RNN architecture designed to handle long-term dependencies.
3.  **Gated Recurrent Unit (GRU):** A simplified version of the LSTM with fewer parameters.
4.  **Bidirectional LSTM (Bi-LSTM):** An extension of LSTM that processes text in both forward and backward directions to capture a richer context.

## Performance Results

The performance of each model was evaluated using test accuracy and a detailed classification report.

### Dataset 1 Results

The models showed moderate performance, with GRU achieving the highest accuracy. The classification reports indicated a significant challenge in classifying minority classes like "Statistics" and "Quantitative Biology," likely due to class imbalance.

| Model     | Test Accuracy |
| :-------- | :------------ |
| **RNN**   | 73.50%        |
| **LSTM**  | 75.93%        |
| **GRU**   | **76.17%**    |
| **Bi-LSTM**| 74.93%       |

### Dataset 2 Results

The models achieved outstanding performance on the second dataset. The combination of advanced preprocessing (lemmatization) and handling of class imbalance (using class weights) led to a dramatic improvement in accuracy across all models.

| Model     | Test Accuracy |
| :-------- | :------------ |
| **LSTM**  | **99.24%**    |
| **GRU**   | 98.98%        |
| **RNN**   | 97.20%        |
| **Bi-LSTM**| **99.24%**    |

## Analysis and Conclusion

*   **Model Architecture:** For both datasets, LSTM and GRU models consistently outperformed the SimpleRNN, demonstrating their superior ability to capture contextual information in text. The Bi-LSTM performed exceptionally well on the second dataset, confirming that processing text in both directions is highly effective.
*   **Impact of Preprocessing:** The significant difference in performance between the two datasets highlights the importance of a robust preprocessing pipeline. The use of spaCy for lemmatization and the application of class weights in the second experiment were critical factors in achieving near-perfect accuracy.
*   **Class Imbalance:** The results from the first dataset underscore the challenges posed by imbalanced data. Without proper handling, models tend to be biased toward the majority classes, resulting in poor performance on minority classes.

## How to Run the Code

1.  **Prerequisites:** Ensure you have Python installed, along with the necessary libraries. You can install them using pip:
    ```
    pip install pandas numpy tensorflow scikit-learn spacy nltk tqdm
    ```
2.  **Download spaCy Model:** Download the English language model for spaCy.
    ```
    python -m spacy download en_core_web_sm
    ```
3.  **Download NLTK Data:** The notebook may require NLTK data for tokenization. Run the following in a Python interpreter:
    ```
    import nltk
    nltk.download('punkt')
    ```
4.  **Set File Paths:** In the Jupyter Notebook (`text-classificaton.ipynb`), update the file paths for `Dataset-1.xlsx` and `A2D2.xlsx` to match their locations on your local machine.
5.  **Execute Notebook:** Run the cells in the Jupyter Notebook sequentially to reproduce the results.

