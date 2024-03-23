# Predictive Analysis of E-Commerce Product Reviews

## Introduction

This project aims to leverage machine learning techniques to analyze e-commerce product reviews, focusing on predicting the number of people that found a review positive (regression) and whether someone would recommend the product (classification). By examining features such as 'Age', 'Rating', 'Department Name', and 'Class Name', the study identifies which variables significantly impact both regression and classification outcomes.

## Central Research Question

My central research question explores how certain variables ('Age', 'Rating', 'Department Name', and 'Class Name') influence the number of other customers who found a review positive and the likelihood of product recommendation. The project aims to contribute to a better understanding of consumer feedback dynamics, assisting businesses in improving their product offerings and customer satisfaction, as well as understanding the effectiveness of natural language processing in machine learning.

## Methodology

### Data Preprocessing
The dataset undergoes several preprocessing steps:

- Dropping rows with missing 'Review Text' and 'Division Name'.
- Imputation of missing values, with a specific focus on using RAKE (Rapid Automatic Keyword Extraction) for extracting keywords from 'Review Text' to impute titles.

### Feature Engineering and Modeling
- The dataset's features, including 'Age', 'Rating', 'Department Name', 'Class Name', and preprocessed 'Review Text', were used to train both regression and classification models.
- Models implemented include Linear Regression, Random Forest Classifier, Random Forest Regressor, and KNeighborsRegressor.
- Language features were extracted using the Count Vectorizer method, aiming to incorporate the semantic content of reviews into the modeling process.

### Challenges Encountered
- Incorporating language learning through Count Vectorizer initially decreased model accuracy. This counterintuitive result is attributed to the introduction of high-dimensional, sparse data from text features, potentially complicating model learning and generalization decreasing accuracy.
- 13.2% of Titles were missing from each row for each review, yet, due to the fact they were user generated, these were very hard to impute, yet too important to delete. Because of this, I taught myself to use RAKE to extract keywords from the user generated reviews, scoring each extracted keyword, taking the highest scored one, and inserting it as the title. This allowed me to impute titles accurately using machine learning models.

## Results

#### - Regression and Classification Impact: The analysis identified 'Rating' as a critical factor for classification outcomes (analyzed through feature importance), whereas 'Class Name' significantly influenced regression results (analyzed through coefficients), although regression results were lackluster, and had high variability in residuals with no pattern.
#### - Model Performance: The inclusion of text data through Count Vectorizer unexpectedly reduced the accuracy of the models. This was likely due to the 'curse of dimensionality', where the models struggled with the complexity and sparsity of text-based features.
#### - Visual Analysis: Residual plots were generated to visualize the differences between predicted and actual values, providing insights into model performance and potential biases, showing that our regression did not fit the data well.
Conclusion

The project demonstrates the nuanced role of various features in predicting the outcomes of product reviews. While textual analysis presents promising avenues for enriching predictive models, it also introduces challenges that necessitate careful feature selection and model tuning. Our findings highlight the importance of understanding the trade-offs between adding complex features and maintaining model accuracy and interpretability.
