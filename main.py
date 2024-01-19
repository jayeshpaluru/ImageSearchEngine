import os
import base64
import numpy as np
import faiss
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from io import BytesIO
import pickle

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to generate embeddings
def generate_embeddings(model, images, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_t = torch.stack([transform(img) for img in batch])
        with torch.no_grad():
            emb = model(batch_t).numpy()
        embeddings.append(emb)

    return np.vstack(embeddings)

# Decode base64 to Image
def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

# Main function
def main():
    # Load pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    model.eval()

    # Load images from dataset
    folder = "/Users/jayeshpaluru/Downloads/Boredom/ImagesForEngine"
    images, filenames = load_images_from_folder(folder)

    # Check if embeddings and index are cached
    if os.path.exists("embeddings.pkl") and os.path.exists("index.faiss"):
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        index = faiss.read_index("index.faiss")
    else:
        # Generate embeddings
        embeddings = generate_embeddings(model, images)

        # Build the Faiss index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Cache embeddings and index
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        faiss.write_index(index, "index.faiss")

    # Search for similar images
    base64_string = input("Enter base64 string of query image: ")
    query_image = decode_base64_to_image(base64_string)
    query_embedding = generate_embeddings(model, [query_image])

    D, I = index.search(query_embedding, k=5)  # Search for top 5 similar images
    print("Top 5 similar images:")
    for i, idx in enumerate(I[0]):
        print(f"Rank {i+1}: {filenames[idx]}, Distance: {D[0][i]}")

if __name__ == "__main__":
    main()
