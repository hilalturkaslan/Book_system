import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv("books.csv")
    return df[['book_id', 'title', 'description']].dropna()


def train_tfidf_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    return tfidf_matrix, vectorizer

def get_recommendations(book_title, df, tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df[df['title'] == book_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]  # En benzer 5 kitap
    book_indices = [i[0] for i in similarity_scores]
    return df.iloc[book_indices][['title', 'description']]

def interactive_recommendation():
    df = load_data()
    tfidf_matrix, _ = train_tfidf_model(df)
    book_title = input("Enter a book title: ")
    if book_title not in df['title'].values:
        print("Book not found. Try another title.")
    else:
        recommendations = get_recommendations(book_title, df, tfidf_matrix)
        print(recommendations)

if __name__ == "__main__":
    interactive_recommendation()

