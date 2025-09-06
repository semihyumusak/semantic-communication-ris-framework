# semantic/semantic_metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


class SemanticMetrics:
    """
    A class implementing various semantic similarity metrics for different data types.
    This extends beyond simple bit error rate to capture true semantic meaning.
    """

    def __init__(self, device="cpu"):
        """
        Initialize semantic metrics calculators.

        Args:
            device (str): Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        # Initialize text models if needed
        self._init_text_model()
        # Initialize image models if needed
        self._init_image_model()

    def _init_text_model(self):
        """Initialize BERT model for text semantic analysis"""
        try:
            self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_model = BertModel.from_pretrained('bert-base-uncased')
            self.text_model.to(self.device)
            self.text_model.eval()
            self.text_model_loaded = True
        except Exception as e:
            print(f"Warning: Text model initialization failed: {e}")
            self.text_model_loaded = False

    def _init_image_model(self):
        """Initialize ResNet model for image semantic analysis"""
        try:
            self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Remove the classification layer to get embeddings
            self.image_model = torch.nn.Sequential(*(list(self.image_model.children())[:-1]))
            self.image_model.to(self.device)
            self.image_model.eval()
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.image_model_loaded = True
        except Exception as e:
            print(f"Warning: Image model initialization failed: {e}")
            self.image_model_loaded = False

    def bit_semantic_accuracy(self, original_bits, recovered_bits):
        """
        Compute traditional semantic accuracy as 1-BER (for baseline comparison).

        Args:
            original_bits (torch.Tensor): Ground truth bits
            recovered_bits (torch.Tensor): Recovered bits

        Returns:
            float: Semantic accuracy (1-BER)
        """
        if not isinstance(original_bits, torch.Tensor):
            original_bits = torch.tensor(original_bits)
        if not isinstance(recovered_bits, torch.Tensor):
            recovered_bits = torch.tensor(recovered_bits)

        total_bits = original_bits.numel()
        bit_errors = (original_bits != recovered_bits).sum().item()
        ber = bit_errors / total_bits
        return 1.0 - ber

    def weighted_bit_semantic_accuracy(self, original_bits, recovered_bits, importance_weights=None):
        """
        Compute weighted semantic accuracy where some bits are more important than others.

        Args:
            original_bits (torch.Tensor): Ground truth bits
            recovered_bits (torch.Tensor): Recovered bits
            importance_weights (torch.Tensor): Weights for each bit's importance (optional)

        Returns:
            float: Weighted semantic accuracy
        """
        if importance_weights is None:
            # Default to uniform weights
            return self.bit_semantic_accuracy(original_bits, recovered_bits)

        if not isinstance(original_bits, torch.Tensor):
            original_bits = torch.tensor(original_bits)
        if not isinstance(recovered_bits, torch.Tensor):
            recovered_bits = torch.tensor(recovered_bits)
        if not isinstance(importance_weights, torch.Tensor):
            importance_weights = torch.tensor(importance_weights)

        # Normalize weights
        importance_weights = importance_weights / importance_weights.sum()

        # Calculate weighted error
        bit_errors = (original_bits != recovered_bits).float()
        weighted_error = (bit_errors * importance_weights).sum().item()

        return 1.0 - weighted_error

    def cosine_similarity(self, original_features, recovered_features):
        """
        Compute cosine similarity between semantic feature vectors.

        Args:
            original_features (torch.Tensor): Original semantic features
            recovered_features (torch.Tensor): Reconstructed semantic features

        Returns:
            float: Cosine similarity score (0 to 1)
        """
        if not isinstance(original_features, torch.Tensor):
            original_features = torch.tensor(original_features, dtype=torch.float32)
        if not isinstance(recovered_features, torch.Tensor):
            recovered_features = torch.tensor(recovered_features, dtype=torch.float32)

        # Ensure vectors are normalized to prevent scaling issues
        original_norm = F.normalize(original_features, p=2, dim=1)
        recovered_norm = F.normalize(recovered_features, p=2, dim=1)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(original_norm, recovered_norm).mean().item()

        # Map from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2

    def text_semantic_similarity(self, original_text, recovered_text):
        """
        Compute semantic similarity between original and recovered text using BERT embeddings.

        Args:
            original_text (str): Original text
            recovered_text (str): Recovered text

        Returns:
            float: Semantic similarity score (0 to 1)
        """
        if not self.text_model_loaded:
            print("Warning: Text model not loaded. Returning BERT-based similarity.")
            return 0.0

        with torch.no_grad():
            # Tokenize texts
            original_tokens = self.text_tokenizer(original_text, return_tensors="pt", padding=True, truncation=True).to(
                self.device)
            recovered_tokens = self.text_tokenizer(recovered_text, return_tensors="pt", padding=True,
                                                   truncation=True).to(self.device)

            # Get BERT embeddings
            original_embedding = self.text_model(**original_tokens).last_hidden_state.mean(dim=1)
            recovered_embedding = self.text_model(**recovered_tokens).last_hidden_state.mean(dim=1)

            # Compute cosine similarity
            similarity = F.cosine_similarity(original_embedding, recovered_embedding).item()

            # Map from [-1, 1] to [0, 1]
            return (similarity + 1) / 2

    def bleu_score(self, reference, hypothesis):
        """
        Compute BLEU score for text similarity

        Args:
            reference (str): Reference (original) text
            hypothesis (str): Hypothesis (recovered) text

        Returns:
            float: BLEU score (0 to 1)
        """
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize

        try:
            reference_tokens = [word_tokenize(reference.lower())]
            hypothesis_tokens = word_tokenize(hypothesis.lower())

            # Calculate BLEU score with equal weights for n-grams
            weights = (0.25, 0.25, 0.25, 0.25)
            score = sentence_bleu(reference_tokens, hypothesis_tokens, weights=weights)
            return score
        except Exception as e:
            print(f"Warning: BLEU score calculation failed: {e}")
            return 0.0

    def image_semantic_similarity(self, original_image, recovered_image):
        """
        Compute semantic similarity between original and recovered images using ResNet features.

        Args:
            original_image (PIL.Image): Original image
            recovered_image (PIL.Image): Recovered image

        Returns:
            float: Semantic similarity score (0 to 1)
        """
        if not self.image_model_loaded:
            print("Warning: Image model not loaded. Returning placeholder similarity.")
            return 0.0

        with torch.no_grad():
            # Preprocess images
            original_tensor = self.image_transform(original_image).unsqueeze(0).to(self.device)
            recovered_tensor = self.image_transform(recovered_image).unsqueeze(0).to(self.device)

            # Extract features
            original_features = self.image_model(original_tensor).flatten(1)
            recovered_features = self.image_model(recovered_tensor).flatten(1)

            # Compute cosine similarity
            similarity = F.cosine_similarity(original_features, recovered_features).item()

            # Map from [-1, 1] to [0, 1]
            return (similarity + 1) / 2

    def structural_similarity(self, original_image, recovered_image, grayscale=True):
        """
        Compute Structural Similarity Index (SSIM) between images

        Args:
            original_image (PIL.Image or numpy array): Original image
            recovered_image (PIL.Image or numpy array): Recovered image
            grayscale (bool): Whether to convert images to grayscale

        Returns:
            float: SSIM score (0 to 1)
        """
        from skimage.metrics import structural_similarity
        import numpy as np

        try:
            # Convert PIL images to numpy arrays if needed
            if isinstance(original_image, Image.Image):
                if grayscale:
                    original_image = original_image.convert('L')
                original_image = np.array(original_image)

            if isinstance(recovered_image, Image.Image):
                if grayscale:
                    recovered_image = recovered_image.convert('L')
                recovered_image = np.array(recovered_image)

            # Calculate SSIM
            score = structural_similarity(
                original_image, recovered_image,
                data_range=original_image.max() - original_image.min(),
                multichannel=not grayscale,
                channel_axis=-1 if not grayscale else None
            )
            return score
        except Exception as e:
            print(f"Warning: SSIM calculation failed: {e}")
            return 0.0

    def mutual_information(self, original_features, recovered_features, bins=20):
        """
        Estimate mutual information between feature vectors using binning

        Args:
            original_features (torch.Tensor or numpy array): Original features
            recovered_features (torch.Tensor or numpy array): Recovered features
            bins (int): Number of bins for histogram

        Returns:
            float: Normalized mutual information score (0 to 1)
        """
        from sklearn.metrics import mutual_info_score

        # Convert to numpy arrays if needed
        if isinstance(original_features, torch.Tensor):
            original_features = original_features.detach().cpu().numpy()
        if isinstance(recovered_features, torch.Tensor):
            recovered_features = recovered_features.detach().cpu().numpy()

        try:
            # Flatten if multi-dimensional
            original_flat = original_features.flatten()
            recovered_flat = recovered_features.flatten()

            # Discretize the features
            original_binned = np.digitize(original_flat, np.linspace(
                original_flat.min(), original_flat.max(), bins))
            recovered_binned = np.digitize(recovered_flat, np.linspace(
                recovered_flat.min(), recovered_flat.max(), bins))

            # Calculate mutual information
            mi = mutual_info_score(original_binned, recovered_binned)

            # Normalize by entropy
            from scipy.stats import entropy
            h1 = entropy(np.bincount(original_binned, minlength=bins + 1) / len(original_binned))
            h2 = entropy(np.bincount(recovered_binned, minlength=bins + 1) / len(recovered_binned))

            if h1 == 0 or h2 == 0:
                return 0.0

            normalized_mi = mi / np.sqrt(h1 * h2)
            return normalized_mi
        except Exception as e:
            print(f"Warning: Mutual information calculation failed: {e}")
            return 0.0

    def evaluate_all_metrics(self, original_bits, recovered_bits, original_features=None,
                             recovered_features=None, feature_importance=None):
        """
        Compute multiple semantic metrics and return as a dictionary

        Args:
            original_bits (torch.Tensor): Original bit sequence
            recovered_bits (torch.Tensor): Recovered bit sequence
            original_features (torch.Tensor, optional): Original semantic features
            recovered_features (torch.Tensor, optional): Recovered semantic features
            feature_importance (torch.Tensor, optional): Feature importance weights

        Returns:
            dict: Dictionary containing multiple semantic metrics
        """
        results = {
            "bit_semantic_accuracy": self.bit_semantic_accuracy(original_bits, recovered_bits)
        }

        # Add weighted semantic accuracy if importance weights provided
        if feature_importance is not None:
            results["weighted_semantic_accuracy"] = self.weighted_bit_semantic_accuracy(
                original_bits, recovered_bits, feature_importance
            )

        # Add feature-based metrics if features provided
        if original_features is not None and recovered_features is not None:
            results["cosine_similarity"] = self.cosine_similarity(
                original_features, recovered_features
            )
            results["mutual_information"] = self.mutual_information(
                original_features, recovered_features
            )

        return results