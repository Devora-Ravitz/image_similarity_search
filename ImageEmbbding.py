import torch
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from PIL import Image

class Img2Vec:
    def __init__(self, model_name='resnet50', weights="imagenet"):
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "vgg19": self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier,
        }
        self.architecture = self.validate_model(model_name)
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.device = self.set_device()
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.dataset = {}
        self.image_clusters = {}
        self.cluster_centers = {}

    def validate_model(self, model_name):
        if model_name not in self.embed_dict.keys():
            raise ValueError(f"The model {model_name} is not supported")
        else:
            return model_name

    def assign_transform(self, weights):
        weights_dict = {
            "resnet50": models.resnet50,
            "vgg19": models.vgg19,
            "efficientnet_b0": models.efficientnet_b0,
        }
        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms
        except Exception:
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return preprocess

    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device

    def initiate_model(self):
        m = getattr(models, self.architecture)
        model = m(pretrained=self.weights == "imagenet")
        model.to(self.device)
        return model.eval()

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()
        return model_embed

    def obtain_children(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])
        return model_embed

    def obtain_classifier(self):
        self.model.fc = nn.Sequential(*list(self.model.fc.children())[:-1])
        return self.model

    def directory_to_list(self, dir):
        ext = (".png", ".jpg", ".jpeg")
        d = os.listdir(dir)
        source_list = [os.path.join(dir, f) for f in d if os.path.splitext(f)[1] in ext]
        return source_list

    def validate_source(self, source):
        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            source_list = self.directory_to_list(source)
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')
        return source_list

    def embed_image(self, img):
        try:
            img = Image.open(img)
            img_trans = self.transform(img)
            if self.device == "cuda:0":
                img_trans = img_trans.cuda()
            img_trans = img_trans.unsqueeze(0)
            return self.embed(img_trans).cpu().detach().numpy()
        except Exception as e:
            print(f"Error embedding image {img}: {e}")
            raise

    def embed_dataset(self, source):
        self.files = self.validate_source(source)
        for file in self.files:
            try:
                vector = self.embed_image(file)
                self.dataset[str(file)] = vector
            except Exception as e:
                print(f"Error embedding {file}: {e}")

    def similar_images(self, target_file, n=None):
        try:
            target_vector = self.embed_image(target_file)
            cosine = nn.CosineSimilarity(dim=1)
            sim_dict = {}
            for k, v in self.dataset.items():
                v_tensor = torch.from_numpy(v)
                target_tensor = torch.from_numpy(target_vector)
                sim = cosine(v_tensor, target_tensor)[0].item()
                sim_dict[k] = sim
            items = sim_dict.items()
            sim_dict = {k: v for k, v in sorted(items, key=lambda i: i[1], reverse=True)}
            if n is not None:
                sim_dict = dict(list(sim_dict.items())[: int(n)])
            self.output_images(sim_dict, target_file)
            return sim_dict
        except Exception as e:
            print(f"Error finding similar images: {e}")

    def find_knn(self, target_file, k=5, deterministic=True):
        try:
            target_vector = self.embed_image(target_file).reshape(1, -1)
            embeddings = [v.reshape(1, -1) for v in self.dataset.values()]
            if len(embeddings) > 0:
                embeddings = np.vstack(embeddings)
                knn = NearestNeighbors(n_neighbors=k, metric='cosine')
                knn.fit(embeddings)
                distances, indices = knn.kneighbors(target_vector)

                if deterministic:
                    similar_images = {}
                    for i, index in enumerate(indices[0]):
                        similar_images[list(self.dataset.keys())[index]] = 1 - distances[0][i]
                else:
                    probabilities = np.exp(-distances) / np.exp(-distances).sum()
                    sampled_index = np.random.choice(indices[0], p=probabilities[0])
                    similar_images = {list(self.dataset.keys())[sampled_index]: 1 - distances[0][
                        np.where(indices[0] == sampled_index)[0][0]]}

                self.output_images(similar_images, target_file)
                return similar_images
            else:
                print("No embeddings found in dataset.")
        except Exception as e:
            print(f"Error finding k-nearest neighbors: {e}")

    def output_images(self, similar, target):
        self.display_img(target, "original")
        for k, v in similar.items():
            self.display_img(k, "similarity:" + str(v))

    def display_img(self, path, title):
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.title(title)
        plt.show()

    def save_dataset(self, path):
        try:
            data = {"model": self.architecture, "embeddings": self.dataset}
            torch.save(data, os.path.join(path, "tensors.pt"))
            print(f"Dataset saved to {os.path.join(path, 'tensors.pt')}")
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def load_dataset(self, source):
        try:
            data = torch.load(source)
            if data["model"] == self.architecture:
                self.dataset = data["embeddings"]
            else:
                raise AttributeError(
                    f'NN architecture "{self.architecture}" does not match the '
                    + f'"{data["model"]}" model used to generate saved embeddings.'
                    + " Re-initiate Img2Vec with correct architecture and reload."
                )
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def cluster_dataset(self, nclusters=6, display=True):
        if not self.dataset:
            print("Dataset is empty. Please embed images first.")
            return

        embeddings = np.array(list(self.dataset.values())).squeeze()
        kmeans = KMeans(n_clusters=nclusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        for i, (image_path, label) in enumerate(zip(self.dataset.keys(), cluster_labels)):
            if label not in self.image_clusters:
                self.image_clusters[label] = {"images": [], "center": None}
            self.image_clusters[label]["images"].append(image_path)

        self.cluster_centers = kmeans.cluster_centers_

        if display:
            self.display_clusters()

    def plot_list(self, img_list, cluster_num):
        fig, axes = plt.subplots(math.ceil(len(img_list) / 2), 2)
        fig.suptitle(f"Cluster: {str(cluster_num)}")
        [ax.axis("off") for ax in axes.ravel()]
        for img, ax in zip(img_list, axes.ravel()):
            ax.imshow(Image.open(img))
        fig.tight_layout()

    def display_clusters(self):
        for num, cluster in self.image_clusters.items():
            img_list = cluster["images"]
            self.plot_list(img_list, num)

if __name__ == "__main__":
    try:
        ImgSim = Img2Vec(model_name='resnet50', weights='imagenet')
        dataset_path = r"C:\Users\326351160\PycharmProjects\pythonProject\dats"
        save_path = r"C:\Users\326351160\PycharmProjects\pythonProject\save"

        # Embedding dataset
        print(f"Embedding dataset from: {dataset_path}")
        ImgSim.embed_dataset(dataset_path)
        ImgSim.save_dataset(save_path)

        # Loading dataset
        print(f"Loading dataset from: {save_path}")
        ImgSim.load_dataset(os.path.join(save_path, "tensors.pt"))

        # Finding similar images
        target_image_path = r"C:\Users\326351160\PycharmProjects\pythonProject\dats\$RZ0Z344.jpg"
        print(f"Finding similar images to: {target_image_path}")
        similar_images = ImgSim.similar_images(target_image_path, n=5)
        print("Similar Images:", similar_images)

        # Finding k-nearest neighbors with deterministic approach
        print(f"Finding k-nearest neighbors for: {target_image_path} (deterministic)")
        knn_images_deterministic = ImgSim.find_knn(target_image_path, k=5, deterministic=True)
        print("K-Nearest Neighbors (deterministic):", knn_images_deterministic)

        # Finding k-nearest neighbors with probabilistic sampling
        print(f"Finding k-nearest neighbors for: {target_image_path} (probabilistic)")
        knn_images_probabilistic = ImgSim.find_knn(target_image_path, k=5, deterministic=False)
        print("K-Nearest Neighbors (probabilistic):", knn_images_probabilistic)

        # Displaying clusters
        print("Clustering dataset")
        ImgSim.cluster_dataset(nclusters=6, display=True)
    except Exception as e:
        print(f"Error: {e}")

