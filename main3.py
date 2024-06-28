import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

# Define the text embedding function using BERT
class TextEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt')
        outputs = self.model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1)

# Define the first autoencoder
class Autoencoder1(nn.Module):
    def __init__(self, input_dim, joint_dim):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Linear(input_dim, joint_dim)
        self.decoder = nn.Linear(joint_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the second autoencoder
class Autoencoder2(nn.Module):
    def __init__(self, input_dim, joint_dim):
        super(Autoencoder2, self).__init__()
        self.encoder = nn.Linear(input_dim, joint_dim)
        self.decoder = nn.Linear(joint_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define JEPA Predictor with input_dim as 1024
class JEPA_Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(JEPA_Predictor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, joint_embedding):
        prediction = self.fc(joint_embedding)
        return prediction

def visualize_weights_as_features(model, tokenizer, sentences1, sentences2, title, n_components=10):
    # Concatenate sentences
    sentences = sentences1 + sentences2
    
    # Get embeddings for the sentences
    embeddings = torch.cat([embedder.get_embedding(s) for s in sentences])

    # Get joint embeddings from both autoencoders
    joint1, _ = autoencoder1(embeddings)
    joint2, _ = autoencoder2(embeddings)

    # Concatenate joint embeddings
    combined_joint = torch.cat((joint1, joint2), dim=1)

    # Forward pass through JEPA predictor
    predictions = model(combined_joint)

    # Convert joint1 to numpy for PCA
    joint1_np = joint1.detach().numpy()

    # Reduce dimensionality with PCA
    pca = PCA(n_components=n_components)
    reduced_joint1 = pca.fit_transform(joint1_np)

    # Create a DataFrame for better visualization
    pc_columns = [f'PC {i+1}' for i in range(n_components)]
    df = pd.DataFrame(reduced_joint1, columns=pc_columns, index=sentences)
    
    # Add a column to indicate the source list (sentences1 or sentences2)
    df['Source'] = ['Sentences 1'] * len(sentences1) + ['Sentences 2'] * len(sentences2)

    # Plot heatmap with seaborn for better control over colors
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[pc_columns], cmap='viridis', annot=False, cbar=True, linewidths=0.5, linecolor='lightgray')
    
    # Color bars based on the source list
    for i, source in enumerate(df['Source'].unique()):
        plt.axhline(len(sentences1) if source == 'Sentences 1' else len(sentences1) + len(sentences2), color='black', lw=2)
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Principal Components')
    plt.ylabel('Sentences')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Prepare example sentences
    sentences1 = [
        "Climate change is significantly altering global weather patterns and ecosystems, causing widespread disruptions.",
        "While rising global temperatures are melting polar ice caps and glaciers, this process is not as rapid or catastrophic as some claim.",
        "Extreme weather events like hurricanes and heatwaves are becoming increasingly frequent and severe due to climate change.",
        "Governments worldwide are finally implementing robust policies to reduce greenhouse gas emissions, which is essential.",
        "Scientists overwhelmingly agree that human activities, especially burning fossil fuels, are the major contributors to climate change.",
        "Climate change mitigation efforts are critical and require international cooperation and binding agreements.",
        "Sea level rise poses a severe threat to coastal communities due to the impacts of climate change.",
        "Climate change adaptation strategies must include building resilient infrastructure to protect vulnerable populations."
    ]

    sentences2 = [
        "Climate change is altering global weather patterns and ecosystems, but the extent and causes are often exaggerated.",
        "Rising global temperatures are melting polar ice caps and glaciers, leading to alarming sea-level rise.",
        "Extreme weather events are becoming more frequent, but attributing them solely to climate change oversimplifies complex natural processes.",
        "Governments worldwide are implementing policies to reduce greenhouse gas emissions, but these measures are often economically burdensome.",
        "While many scientists agree on human contributions to climate change, the extent of this impact is still debated.",
        "Climate change mitigation efforts are necessary but should not compromise national sovereignty and economic growth.",
        "Sea level rise threatens coastal communities, but adaptive strategies and technological innovations can mitigate these risks effectively.",
        "Climate change adaptation strategies include building resilient infrastructure, but funding and priorities should be carefully managed."
    ]

    # Initialize the text embedder
    embedder = TextEmbedder()

    # Define dimensions
    input_dim = 768  # BERT embedding dimension
    joint_dim = 512  # Joint embedding dimension

    # Initialize the autoencoders
    autoencoder1 = Autoencoder1(input_dim, joint_dim)
    autoencoder2 = Autoencoder2(input_dim, joint_dim)

    # Initialize the JEPA predictor
    predictor_input_dim = 2 * joint_dim  # Concatenated joint embeddings from both autoencoders
    predictor_output_dim = 2  # Dimension of the predicted output
    predictor = JEPA_Predictor(predictor_input_dim, predictor_output_dim)

    # Define optimizer for the predictor
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=1e-3)

    # Training loop for autoencoders (unsupervised)
    num_epochs_ae = 10
    for epoch in range(num_epochs_ae):
        total_loss_ae1 = 0
        total_loss_ae2 = 0
        for sentence1, sentence2 in zip(sentences1, sentences2):
            embedding1 = embedder.get_embedding(sentence1)
            embedding2 = embedder.get_embedding(sentence2)

            # Autoencoder 1
            encoded1, decoded1 = autoencoder1(embedding1)
            loss_ae1 = nn.MSELoss()(decoded1, embedding1)
            optimizer_ae1 = optim.Adam(autoencoder1.parameters(), lr=1e-3)
            optimizer_ae1.zero_grad()
            loss_ae1.backward()
            optimizer_ae1.step()
            total_loss_ae1 += loss_ae1.item()

            # Autoencoder 2
            encoded2, decoded2 = autoencoder2(embedding2)
            loss_ae2 = nn.MSELoss()(decoded2, embedding2)
            optimizer_ae2 = optim.Adam(autoencoder2.parameters(), lr=1e-3)
            optimizer_ae2.zero_grad()
            loss_ae2.backward()
            optimizer_ae2.step()
            total_loss_ae2 += loss_ae2.item()

        print(f"Autoencoder 1 - Epoch {epoch + 1}/{num_epochs_ae}, Loss: {total_loss_ae1}")
        print(f"Autoencoder 2 - Epoch {epoch + 1}/{num_epochs_ae}, Loss: {total_loss_ae2}")

   # Training loop for JEPA Predictor (supervised)
    num_epochs_predictor = 10
    for epoch in range(num_epochs_predictor):
        total_loss_predictor = 0
        for sentence1, sentence2 in zip(sentences1, sentences2):
            embedding1 = embedder.get_embedding(sentence1)
            embedding2 = embedder.get_embedding(sentence2)

            # Get joint embeddings from both autoencoders
            joint1, _ = autoencoder1(embedding1)
            joint2, _ = autoencoder2(embedding2)

            # Concatenate joint embeddings
            combined_joint = torch.cat((joint1, joint2), dim=1)

            # Forward pass through JEPA predictor
            prediction = predictor(combined_joint)

            # Determine target based on similarity of sentences (adjust according to your task)
            if sentence1 == sentence2:
                target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # Similar sentences
            else:
                target = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # Different sentences

            # Calculate loss (MSE loss)
            loss_predictor = nn.MSELoss()(prediction, target)

            # Backward pass and optimization
            optimizer_predictor.zero_grad()
            loss_predictor.backward()
            optimizer_predictor.step()

            total_loss_predictor += loss_predictor.item()

        print(f"JEPA Predictor - Epoch {epoch + 1}/{num_epochs_predictor}, Loss: {total_loss_predictor}")


    # Save models
    torch.save(autoencoder1.state_dict(), "autoencoder1.pth")
    torch.save(autoencoder2.state_dict(), "autoencoder2.pth")
    torch.save(predictor.state_dict(), "predictor.pth")

    # Visualize weights as features in word space
    visualize_weights_as_features(predictor, embedder.tokenizer, sentences1, sentences2, "Weights as Features in Word Space")
