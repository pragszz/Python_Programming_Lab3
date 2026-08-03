import numpy as np
from pathlib import Path
import json
from embeddings import get_embedding
import matplotlib.pyplot as plt

DOCUMENTS_FILE = "documents.json"

# Documents.json file is constructed (load, embed)
def load_documents(path=DOCUMENTS_FILE):
    with open(path, "r", encoding="utf-8") as f:
            print(path)
            return json.load(f)

def build_embedding_matrix(documents):
    raw_doc = load_documents(DOCUMENTS_FILE)
    embeddings = []
    for doc in raw_doc:
        text = doc["text"]
        embedding = get_embedding(text, input_type="passage")["data"][0]["embedding"]
        embeddings.append(embedding)

    embedding_matrix = np.stack(embeddings, axis =0)
    return embedding_matrix

# Cosine Similarity
def search(query, embedding_matrix, documents, top_k):

    distances = []
    query_embedding = get_embedding(query, input_type="query")["data"][0]["embedding"]
    print(query_embedding)
    for i in embedding_matrix:
        distance = cosine_similarity(query_embedding, i)
        distances.append(distance)

    k_indices = np.argsort(distances)[::-1][:top_k] 

    return [documents[i] for i in k_indices]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pca_via_svd(X_data, n_components):
    centred_data = X_data - X_data.mean(axis=0)
    U, S, Vt = np.linalg.svd(centred_data, full_matrices=False)

    principal_components = Vt[:n_components]

    X_projected = centred_data @ principal_components.T

    print(U.shape, S.shape, Vt.shape)
    return principal_components, X_projected, centred_data

def plot_grid(X_projected, documents):
    fig, ax = plt.subplots(figsize=(8, 6))

    topics = []
    for doc in documents:
        if doc["topic"] not in topics:
            topics.append(doc["topic"])

    for topic in topics:
        # Find indices of documents matching this topic
        indices = []
        for i, doc in enumerate(documents):
            if doc["topic"] == topic:
                indices.append(i)

        ax.scatter(
            X_projected[indices, 0],
            X_projected[indices, 1],
            label=topic,
            alpha=0.7,
            s=40
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Document Embeddings Projected via PCA (2D)")
    ax.legend()
    ax.axis("equal")
    plt.show()

#Results
if __name__ == "__main__":
    documents = load_documents(DOCUMENTS_FILE)
    embedding_matrix = build_embedding_matrix(documents)
    print(search("What is NVIDIA doing", embedding_matrix, documents, 4))
    principal_components, X_projected, centred_data = pca_via_svd(embedding_matrix, 4)
    plot_grid(X_projected, documents)

