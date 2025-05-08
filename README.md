# Machine-Learning-Model-Implementation
A machine learning model that classifies SMS messages as spam or ham using text vectorization and Multinomial Naive Bayes algorithm.

**COMPANY**: CODETECH IT SOLUTIONS

**Name**: Siddharth Patel

**INTERN ID**: CT04DL819

**Domain**: Python Programming

**BATCH Duration**: 4 Weeks

**Mentor**: Neela Santhosh Kumar

**PROJECT**: MACHINE LEARNING MODEL IMPLEMENTATION

# SMS Spam Classifier 

## Project Overview
This project implements a Naive Bayes machine learning model to classify SMS messages as either spam or legitimate (ham). It demonstrates a complete natural language processing (NLP) pipeline from data preprocessing to model evaluation.

## Key Features
- Text preprocessing and feature extraction using CountVectorizer
- Multinomial Naive Bayes classification model
- Model evaluation using accuracy metrics
- Demonstration of predictions on sample messages

## Technical Implementation

### Data Processing
- Input data format: TSV file with columns (label, message)
- Label encoding: ham=0, spam=1
- Dataset split: 80% training, 20% testing (random_state=42 for reproducibility)

### Feature Engineering
- CountVectorizer converts text messages into numerical features
- Creates a bag-of-words representation of the SMS data
- Handles text tokenization and word frequency counting automatically

### Model Training
- Algorithm: Multinomial Naive Bayes
- Trained on vectorized text features (X_train_counts) and labels (y_train)
- Efficient implementation suitable for text classification tasks

### Evaluation Metrics
- Accuracy score calculated on test set
- Example predictions:
  - "Free prize! Click now" correctly classified as spam
  - "Hi Bob, meeting tomorrow" correctly classified as ham

## How to Use

### Requirements
- Python 3.x
- Required packages: pandas, scikit-learn

Install dependencies with:
```
pip install pandas scikit-learn
```

### Running the Project
1. Clone the repository
2. Run the script:
```
python task4.py
```

##Output:

![Screenshot 2025-05-05 162244](https://github.com/user-attachments/assets/b9713066-5fbe-4b27-94f5-029d125bbef4)


## License
- **This project**: [MIT License](LICENSE) - Free for any use  
