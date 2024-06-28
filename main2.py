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

class TextJEPA(nn.Module):
    def __init__(self, input_dim, joint_dim, output_dim):
        super(TextJEPA, self).__init__()
        self.fc1 = nn.Linear(input_dim, joint_dim)
        self.fc2 = nn.Linear(input_dim, joint_dim)
        self.predictor = nn.Linear(joint_dim, output_dim)

    def forward(self, embedding1, embedding2):
        joint1 = self.fc1(embedding1)
        joint2 = self.fc2(embedding2)
        prediction = self.predictor(joint1)
        return joint1, joint2, prediction

# Define the contrastive loss function
def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    distances = (embedding1 - embedding2).pow(2).sum(1)  # Euclidean distance
    losses = label * distances + (1 - label) * torch.clamp(margin - distances, min=0.0)
    return losses.mean()

# Define the prediction loss function
def prediction_loss(predictions, targets):
    return nn.MSELoss()(predictions, targets)

def visualize_weights_as_features(model, tokenizer, sentences1, sentences2, labels, title, n_components=10):
    # Concatenate sentences
    sentences = sentences1 + sentences2
    
    # Get embeddings for the sentences
    embeddings = torch.cat([embedder.get_embedding(s) for s in sentences])
    joint1, _, _ = model(embeddings, embeddings)

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

# Example usage
if __name__ == "__main__":
    # Prepare example sentences and labels
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

    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)  # 1 if similar, 0 if different
    targets = torch.tensor([
                        [1.0, 0.0],   # Target for the first pair (similar)
                        [1.0, 0.0],   # Target for the second pair (similar)
                        [1.0, 0.0],   # Target for the third pair (similar)
                        [1.0, 0.0],   # Target for the fourth pair (similar)
                        [0.0, 1.0],   # Target for the fifth pair (different)
                        [0.0, 1.0],   # Target for the sixth pair (different)
                        [0.0, 1.0],
                        [0.0, 1.0]
                        ],  # Target for the seventh pair (different)
                        dtype=torch.float32)

    # Initialize the text embedder and model
    embedder = TextEmbedder()
    input_dim = 768  # BERT embedding dimension
    joint_dim = 512  # Joint embedding dimension
    output_dim = 2  # Dimension of the predicted output
    model = TextJEPA(input_dim, joint_dim, output_dim)

    # Get embeddings for the sentences
    embeddings1 = torch.cat([embedder.get_embedding(s) for s in sentences1])
    embeddings2 = torch.cat([embedder.get_embedding(s) for s in sentences2])

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        joint1, joint2, pred1 = model(embeddings1, embeddings2)
        loss1 = contrastive_loss(joint1, joint2, labels)
        loss2 = prediction_loss(pred1, targets)
        loss = loss1 + loss2
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "text_jepa_model.pth")

    # Visualize weights as features in word space
    visualize_weights_as_features(model, embedder.tokenizer, sentences1, sentences2, labels, "Weights as Features in Word Space")
