import json
import torch

import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# design exploration for masks
# this approach was rejected due to an insufficient capacity to learn useful masks
# =============================================================================
# class XSec(nn.Module):
#     def __init__(self, config, device=None):
#         super(XSec, self).__init__()
#         self.device = device if device else torch.device('cpu')
#         num_classes = config["num_classes"]
#         num_prototypes_per_class = config["num_prototypes_per_class"]
#         num_prototypes = num_classes * num_prototypes_per_class

#         feature_dim = config["feature_dim"]
#         embedding_dim = config["embedding_dim"]

#         # define masks
#         self.masks = nn.Parameter(
#             torch.rand((num_prototypes, feature_dim)),
#             requires_grad=True
#         )

#         # define all models
#         self.encoder = encoder_map[config["dataset"]](config)

#         # define prototypes
#         self.prototypes = nn.Parameter(
#             torch.rand((num_prototypes, embedding_dim)),
#             requires_grad=True
#         )

#         # define and set last layer weights
#         self.last_layer = LastLayer(config)

#     def calculate_similarity(self, x):
#         """
#         calculate the similarity scores between prototypes and subfeature embeddings
#         input_shape: (num_samples, num_prototypes, feature_dim)
#         output_shape: (num_samples, num_prototypes)
#         """
#         distances = torch.linalg.vector_norm(x - self.prototypes, ord=2, dim=2) ** 2
#         similarities = torch.log((distances + 1) / (distances + 1e-8))
#         return similarities, distances

#     def forward(self, x):
#         sub_features = x.unsqueeze(1) * self.masks
#         encoded = self.encoder(sub_features)
#         # decoded = self.decoder(encoded)

#         similarities, distances = self.calculate_similarity(encoded)
#         predictions = self.last_layer(similarities)
#         return predictions, distances, similarities, self.masks
# =============================================================================


class XSec(nn.Module):
    def __init__(self, config, device=None):
        super(XSec, self).__init__()
        self.device = device if device else torch.device('cpu')

        num_classes = config["num_classes"]
        num_prototypes_per_class = config["num_prototypes_per_class"]
        num_prototypes = num_classes * num_prototypes_per_class
        embedding_dim = config["embedding_dim"]

        # define all models
        self.mask_generator = mask_gen_map[config["dataset"]](config)
        self.encoder = encoder_map[config["dataset"]](config)

        # define masks
        self.maskgen_input = nn.Parameter(
            torch.rand((num_prototypes,)),
            requires_grad=True
        )

        # define prototypes
        self.prototypes = nn.Parameter(
            torch.rand((num_prototypes, embedding_dim)),
            requires_grad=True
        )

        # define and set last layer weights
        self.last_layer = LastLayer(config)

    def calculate_similarity(self, x):
        """
        calculate the similarity scores between prototypes and subfeature embeddings
        input_shape: (num_samples, num_prototypes, feature_dim)
        output_shape: (num_samples, num_prototypes)
        """
        distances = torch.linalg.vector_norm(x - self.prototypes, ord=2, dim=2) ** 2
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        return distances, similarities

    def forward(self, x):
        masks = self.mask_generator(self.maskgen_input)
        sub_features = x.unsqueeze(1) * masks
        encoded = self.encoder(sub_features)

        distances, similarities = self.calculate_similarity(encoded)
        predictions = self.last_layer(similarities)
        return predictions, distances, similarities, masks

    def predict_proba(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        pred, _, _, _ = self.forward(x)
        pred = F.softmax(pred, dim=1).cpu().detach().numpy()
        return pred


class PDFEncoder(nn.Module):
    def __init__(self, config):
        super(PDFEncoder, self).__init__()
        feature_dim = config["feature_dim"]
        embedding_dim = config["embedding_dim"]
        self.linear1 = nn.Linear(feature_dim, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x


class PhishingEncoder(nn.Module):
    def __init__(self, config):
        super(PhishingEncoder, self).__init__()
        feature_dim = config["feature_dim"]
        embedding_dim = config["embedding_dim"]
        self.linear1 = nn.Linear(feature_dim, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x


class NetflowEncoder(nn.Module):
    def __init__(self, config):
        super(NetflowEncoder, self).__init__()
        feature_dim = config["feature_dim"]
        embedding_dim = config["embedding_dim"]
        self.linear1 = nn.Linear(feature_dim, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x


class BodmasEncoder(nn.Module):
    def __init__(self, config):
        super(BodmasEncoder, self).__init__()
        feature_dim = config["feature_dim"]
        embedding_dim = config["embedding_dim"]
        self.linear1 = nn.Linear(feature_dim, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x


class NSLKDDEncoder(nn.Module):
    def __init__(self, config):
        super(NSLKDDEncoder, self).__init__()
        feature_dim = config["feature_dim"]
        embedding_dim = config["embedding_dim"]
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        batch_size, num_versions, features = x.shape
        x_reshaped = x.reshape(-1, 1, features)
        _, (hidden, _) = self.lstm(x_reshaped)
        output = hidden[-1].reshape(batch_size, num_versions, -1)
        return output


class LastLayer(nn.Module):
    def __init__(self, config):
        super(LastLayer, self).__init__()
        num_classes = config["num_classes"]
        num_prototypes_per_class = config["num_prototypes_per_class"]
        num_prototypes = num_classes * num_prototypes_per_class
        self.linear = nn.Linear(num_prototypes, num_classes, bias=False)
        pos_weight_loc = torch.zeros((num_classes, num_prototypes))
        for c in range(num_classes):
            for i in range(num_prototypes_per_class):
                pos_weight_loc[c, c * num_prototypes_per_class + i] = 1
        neg_weight_loc = 1 - pos_weight_loc
        self.linear.weight.data.copy_(1 * pos_weight_loc - 0.5 * neg_weight_loc)
    
    def forward(self, x):
        return self.linear(x)


class PDFMaskGenerator(nn.Module):
    def __init__(self, config) -> None:
        super(PDFMaskGenerator, self).__init__()
        self.config = config
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * config["num_prototypes_per_class"]
        self.linear1 = nn.Linear(num_prototypes, 512)
        self.linear2 = nn.Linear(512 , 1024)
        self.linear3 = nn.Linear(1024 , 2048)
        self.linear4 = nn.Linear(2048, feature_dim * num_prototypes)

    def forward(self, x):
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * self.config["num_prototypes_per_class"]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        return x.reshape((num_prototypes, feature_dim))


class PhishingMaskGenerator(nn.Module):
    def __init__(self, config) -> None:
        super(PhishingMaskGenerator, self).__init__()
        self.config = config
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * config["num_prototypes_per_class"]
        self.linear1 = nn.Linear(num_prototypes, 256)
        self.linear2 = nn.Linear(256 , 512)
        self.linear3 = nn.Linear(512 , 1024)
        self.linear4 = nn.Linear(1024, feature_dim * num_prototypes)

    def forward(self, x):
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * self.config["num_prototypes_per_class"]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        return x.reshape((num_prototypes, feature_dim))


class NetflowMaskGenerator(nn.Module):
    def __init__(self, config) -> None:
        super(NetflowMaskGenerator, self).__init__()
        self.config = config
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * config["num_prototypes_per_class"]
        self.linear1 = nn.Linear(num_prototypes, 512)
        self.linear2 = nn.Linear(512 , 1024)
        self.linear3 = nn.Linear(1024 , 2048)
        self.linear4 = nn.Linear(2048, feature_dim * num_prototypes)

    def forward(self, x):
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * self.config["num_prototypes_per_class"]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        return x.reshape((num_prototypes, feature_dim))


class BodmasMaskGenerator(nn.Module):
    def __init__(self, config) -> None:
        super(BodmasMaskGenerator, self).__init__()
        self.config = config
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * config["num_prototypes_per_class"]
        self.linear1 = nn.Linear(num_prototypes, 256)
        self.linear2 = nn.Linear(256 , 512)
        self.linear3 = nn.Linear(512 , 1024)
        self.linear4 = nn.Linear(1024, feature_dim * num_prototypes)

    def forward(self, x):
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * self.config["num_prototypes_per_class"]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        return x.reshape((num_prototypes, feature_dim))


class NSLKDDMaskGenerator(nn.Module):
    def __init__(self, config) -> None:
        super(NSLKDDMaskGenerator, self).__init__()
        self.config = config
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * config["num_prototypes_per_class"]
        self.linear1 = nn.Linear(num_prototypes, 512)
        self.linear2 = nn.Linear(512 , 1024)
        self.linear3 = nn.Linear(1024 , 2048)
        self.linear4 = nn.Linear(2048, feature_dim * num_prototypes)

    def forward(self, x):
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * self.config["num_prototypes_per_class"]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        return x.reshape((num_prototypes, feature_dim))


encoder_map = {
    "pdf": PDFEncoder,
    "phishing": PhishingEncoder,
    "netflow": NetflowEncoder,
    "bodmas": BodmasEncoder,
    "bodmas200": BodmasEncoder,
    "bodmas500": BodmasEncoder,
    "nsl_kdd_multi": NSLKDDEncoder,
}

mask_gen_map = {
    "pdf": PDFMaskGenerator,
    "phishing": PhishingMaskGenerator,
    "netflow": NetflowMaskGenerator,
    "bodmas": BodmasMaskGenerator,
    "bodmas200": BodmasMaskGenerator,
    "bodmas500": BodmasMaskGenerator,
    "nsl_kdd_multi": NSLKDDMaskGenerator,
}