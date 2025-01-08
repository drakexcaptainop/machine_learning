from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


descriptions = train_data.name

query = "4K LED Croma"

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(descriptions)

lsa = TruncatedSVD(n_components=70)  # Reduce to 2 dimensions
X_lsa = lsa.fit_transform(X)

query_vec = vectorizer.transform([query])
query_lsa = lsa.transform(query_vec)

similarities = cosine_similarity(query_lsa, X_lsa)

ranked_indices = similarities.argsort()[0][::-1]
print("Ranked Descriptions:")
for idx in ranked_indices:
    print(descriptions[idx])
