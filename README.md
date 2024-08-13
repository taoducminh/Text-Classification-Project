
# Spam Text Classification using Naive Bayes

This project involves building a spam text classification model using the Naive Bayes algorithm. The model is trained on a dataset containing labeled messages (spam or ham) and aims to classify incoming text messages as either spam or non-spam.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

In this project, we implement a Naive Bayes classifier to distinguish between spam and non-spam messages. The key steps involved include:
1. Loading the dataset.
2. Preprocessing the data.
3. Training a Naive Bayes model.
4. Evaluating the model's performance.

## Dataset

The dataset used for this project is a collection of labeled text messages. Each message is categorized as either 'spam' or 'ham' (non-spam). The dataset is publicly available and was downloaded using the `gdown` command. 

**Dataset Link:** [Spam Text Classification Dataset](https://drive.google.com/file/d/1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R/view?usp=sharing)

If you encounter download limits, you can manually upload the dataset to your Google Drive and use the following commands to access it from Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /path/to/dataset/on/your/drive .
```

## Installation

To run the notebook and experiment with the code, you need to install the necessary Python libraries. You can install these dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

## Usage

To use this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-text-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd spam-text-classification
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Spam_Text_Classification_Naive_Bayes.ipynb
   ```
4. Follow the instructions in the notebook to load the dataset, preprocess the data, train the model, and evaluate its performance.

## Model

The Naive Bayes model is implemented using the `GaussianNB` classifier from the `scikit-learn` library. The following steps are performed in the notebook:
- **Data Preprocessing:** Tokenization, stopword removal, and label encoding.
- **Model Training:** Training the Naive Bayes model on the preprocessed data.
- **Prediction:** Using the trained model to classify new messages as spam or non-spam.

## Evaluation

The model's performance is evaluated using accuracy as the metric. The notebook includes code to split the data into training and testing sets, train the model, and calculate the accuracy score on the test set.

## Conclusion

The Naive Bayes classifier provides a simple yet effective solution for spam text classification. The model can be further improved by experimenting with different text preprocessing techniques and model parameters.

## Contributing

Contributions are welcome! If you have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
