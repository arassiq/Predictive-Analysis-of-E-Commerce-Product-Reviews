from rake_nltk import Rake
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/Application of Analytics/Final Project Walmart/Womens Clothing E-Commerce Reviews.csv')

df = df.dropna(subset= ['Review Text'])

sample_reviews = df[df['Title'].isna()]['Review Text'].sample(5, random_state=1)

rake = Rake()

for i, review in enumerate(sample_reviews):
    rake.extract_keywords_from_text(review)
    keywordsWithScores = rake.get_ranked_phrases_with_scores()
    print(f"Review {i+1} Keywords and Scores:\n{keywordsWithScores[:5]}\n")