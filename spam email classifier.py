import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st


# Load the CSV and assign proper column names
data = pd.read_csv("C:\\Users\\hp\\html\\mini projects\\spam data collection.csv", header=None, names=['index', 'category', 'message'], encoding='latin1')
# Drop duplicate rows
data.drop_duplicates(inplace=True)

# Drop the index column as it's not needed
data.drop(columns=['index'], inplace=True)

# Print head and shape for confirmation
#print(data.head())
print(data.shape)

# Convert 'ham'/'spam' to 'not spam'/'spam'
data['category'] = data['category'].replace(['ham', 'spam'], ['not spam', 'spam'])

# Split into features and labels
mess = data['message']
cat = data['category']
print(data.head())
# Train-test split
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

# Vectorize the messages
cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(features_train, cat_train)

# Test accuracy
features_test = cv.transform(mess_test)
print("Model Accuracy:", model.score(features_test, cat_test))

# Prediction function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result[0]

st.header('spam detection')
# Example usage
#print(predict("You've won a free ticket! Text WIN to 12345 now!"))

input_mess=st.text_input('enter message here')
if st.button('validate'):
   output=predict(input_mess)
   st.markdown(output)
   


#import os
#import sys

#if __name__ == "__main__":
    #if not any("streamlit" in arg for arg in sys.argv):
        # Replace below path with your actual streamlit.exe path
        #streamlit_path = r"c:\Users\hp\html\mini projects\spam email classifier.py"

        #os.system(f'"{streamlit_path}" run "{sys.argv[0]}"')
        
        #sys.exit()
