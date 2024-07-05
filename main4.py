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

def visualize_weights_as_features(model, tokenizer, sentences1, sentences2, title):
    '''
    Visualize the importance of the predictor model's weights for the joint embeddings of pairs of sentences. This visualization helps interpret how the model views the relationship between different sentence pairs in terms of their combined embeddings.

    - The values in the heatmap indicate how strongly the predictor model's weights are influenced by the combined joint embedding of each pair of sentences.
    - High values in the heatmap suggest that the predictor's weights have a strong influence on the joint embedding of the corresponding pair of sentences.
    - Low values suggest a weaker influence.
    - The heatmap does not show the actual predictions (i.e., output values) made by the predictor model for each pair of sentences.
    - It does not directly reflect how the model would score or classify each pair of sentences in terms of similarity, relevance, or any other specific output metric.

    Interpretation:

    Model Evaluation:
    - By visualizing the heatmap of importance values, you gain insights into which pairs of sentences the model considers most relevant based on its fixed weights.
    - This helps understand which features or embeddings from the sentences are critical for the model's decision-making process.

    Comparative Analysis:
    - You can compare different pairs of sentences to see how they stack up in terms of importance, identifying patterns or biases in the model's focus.

    Post-Training Analysis:
    - Since the weights are fixed after training, the importance values reflect how well each pair of sentences fits the learned patterns encoded in those weights.
    - This analysis is a reflection of the model's internal representation and the relationships it has learned during training.
    '''
    # Initialize empty matrices to store weights and importance
    num_sentences1 = len(sentences1)
    num_sentences2 = len(sentences2)
    importance_matrix = np.zeros((num_sentences1, num_sentences2))

    # Get the weights from the linear layer of the model
    weights = model.fc.weight.detach().numpy()  # Assuming model.fc is the linear layer

    for i, sentence1 in enumerate(sentences1):
        embedding1 = embedder.get_embedding(sentence1)
        joint1, _ = autoencoder1(embedding1)

        for j, sentence2 in enumerate(sentences2):
            embedding2 = embedder.get_embedding(sentence2)
            joint2, _ = autoencoder2(embedding2)

            # Concatenate joint embeddings
            combined_joint = torch.cat((joint1, joint2), dim=1)

            # computes the dot product between the weight matrix and the combined joint embedding. This dot product essentially measures how much each weight contributes to the joint embedding's importance.
            importance_matrix[i, j] = np.abs(weights @ combined_joint.detach().numpy().flatten())

    # Plotting the heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(importance_matrix, cmap='viridis', annot=True, fmt='.2f',
                xticklabels=sentences2, yticklabels=sentences1, cbar=True)

    # Automatically wrap tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f'{title} - Predictor Weights Importance')
    plt.xlabel('Sentences in set 2')
    plt.ylabel('Sentences in set 1')
    plt.tight_layout()
    plt.show()

def visualize_predictions_as_features(model, tokenizer, sentences1, sentences2, title, n_components=10):
    '''
    Uses JEPA to visualise predictions of model using data it has already seen. 
    '''
    sentences = sentences1 + sentences2
    embeddings = torch.cat([embedder.get_embedding(s) for s in sentences])
    joint1, _ = autoencoder1(embeddings)
    joint2, _ = autoencoder2(embeddings)
    combined_joint = torch.cat((joint1, joint2), dim=1)
    predictions = model(combined_joint)
    
    # Detach and convert predictions to numpy
    predictions_np = predictions.detach().numpy()
    
    # Convert joint1 to numpy for PCA
    joint1_np = joint1.detach().numpy()
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=n_components)
    reduced_joint1 = pca.fit_transform(joint1_np)
    
    # Create a DataFrame for better visualization
    pc_columns = [f'PC {i+1}' for i in range(n_components)]
    df = pd.DataFrame(reduced_joint1, columns=pc_columns, index=sentences)
    df['Source'] = ['Sentences 1'] * len(sentences1) + ['Sentences 2'] * len(sentences2)
    
    # Add predictions to the DataFrame
    df['Predictions'] = predictions_np
    
    # Plot predictions if needed
    plt.figure(figsize=(12, 8))
    sns.barplot(x=df.index, y='Predictions', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{title} - Predictions')
    plt.xlabel('Sentences')
    plt.ylabel('Prediction Value')
    plt.show()


def visualize_new_embeddings_as_features(model, tokenizer, sentences1, sentences2, title, n_components=10):
    '''
    Does not use JEPA. Shows embeddings against corresponding word representations across 10 PCs.
    '''
    # Concatenate sentences
    sentences = sentences1 + sentences2
    
    # Get embeddings for the sentences
    embeddings = torch.cat([embedder.get_embedding(s) for s in sentences])

    # Get joint embeddings from both autoencoders
    joint1, _ = autoencoder1(embeddings)
    joint2, _ = autoencoder2(embeddings)

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
    predictor_output_dim = 1  # Dimension of the predicted output
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

            # Calculate target similarity based on distance (assumed to be similar)
            target_similarity = 1.0 - torch.mean((joint1 - joint2)**2)

            # Calculate loss (MSE loss)
            loss_predictor = nn.MSELoss()(prediction, target_similarity.view(1, 1))

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
    
    #visualize_new_embeddings_as_features(predictor, embedder.tokenizer, sentences1, sentences2, "Weights as Features in Word Space")
    #visualize_predictions_as_features(predictor, embedder.tokenizer, sentences1, sentences2, "Weights as Features in Word Space")

    visualize_weights_as_features(predictor, embedder.tokenizer, sentences1, sentences2, "Weights as Features in Word Space")
