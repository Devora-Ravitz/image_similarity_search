Image Similarity and Clustering with ResNet50

1. Image Processing 🖼️
Resize: The images are resized to 224x224 pixels to meet the model's requirements.
Normalize: Pixel values are normalized to improve model performance and accuracy.
Ready for Model: The clothing images are now ready to be fed into the model for vector extraction.
2. Model Initialization 🧠
ResNet50 Pre-trained Weights: The ResNet50 model, pre-trained on ImageNet, is used to generate embeddings—vectors representing the essential features of the images, facilitating easy comparison.
3. Saving and Loading the Dataset 💾
Backup: It's always good to have a backup. The vectors are saved as a file and can be reloaded for further processing if needed.
4. Finding Similar Clothing 🔍
Cosine Distance: The vector of a given clothing image is compared with vectors in the dataset to find the most similar clothing items.
5. K-Nearest Neighbors 🧑‍🏫
KNN Algorithm: The K-nearest neighbors for a given clothing image are found using the Nearest Neighbors algorithm. This algorithm helps classify or predict the category or value of a new sample based on its similarity to previous samples.
6. Clustering 📊
KMeans Clustering: The clothing images in the dataset are clustered using KMeans, which groups the samples into several clusters, keeping the internal proximity within each cluster.
Visual Representation: The clusters are displayed visually, allowing you to see which images belong to the same cluster.
Installation and Usage
Prerequisites
Python
PyTorch
NumPy
Matplotlib
scikit-learn
PIL (Pillow)
How to Run
Initialize the Model and Dataset:

python
Copy code
ImgSim = Img2Vec(model_name='resnet50', weights='imagenet')
dataset_path = "/path/to/your/dataset"
save_path = "/path/to/save/embeddings"

# Embedding dataset
ImgSim.embed_dataset(dataset_path)
ImgSim.save_dataset(save_path)
Load Dataset:

python
Copy code
ImgSim.load_dataset(os.path.join(save_path, "tensors.pt"))
Find Similar Images:

python
Copy code
target_image_path = "/path/to/your/target_image.jpg"
similar_images = ImgSim.similar_images(target_image_path, n=5)
print("Similar Images:", similar_images)
K-Nearest Neighbors:

python
Copy code
knn_images_deterministic = ImgSim.find_knn(target_image_path, k=5, deterministic=True)
print("K-Nearest Neighbors (deterministic):", knn_images_deterministic)
Clustering:

python
Copy code
ImgSim.cluster_dataset(nclusters=6, display=True)
Code Structure
The main class, Img2Vec, handles all major functionalities:

Initialization and Model Setup
Dataset Embedding
Finding Similar Images
K-Nearest Neighbors
Clustering and Visualization
Conclusion
This project allows you to find similar clothing items and cluster them using pre-trained models and various machine learning techniques. It demonstrates the practical use of deep learning models for image processing and retrieval tasks.

Feel free to customize and expand this README as needed for your project!

תעשה שישאיר את האותיות הגדולות ולא יקטין כשאני מעתיקה
ChatGPT
בבקשה, הנה הגרסה עם אותיות גדולות כראוי. תוכל להעתיק ולהדביק את הטקסט מבלי שהאותיות הגדולות ישתנו.

IMAGE SIMILARITY AND CLUSTERING WITH RESNET50
1. IMAGE PROCESSING 🖼️
RESIZE: THE IMAGES ARE RESIZED TO 224X224 PIXELS TO MEET THE MODEL'S REQUIREMENTS.
NORMALIZE: PIXEL VALUES ARE NORMALIZED TO IMPROVE MODEL PERFORMANCE AND ACCURACY.
READY FOR MODEL: THE CLOTHING IMAGES ARE NOW READY TO BE FED INTO THE MODEL FOR VECTOR EXTRACTION.
2. MODEL INITIALIZATION 🧠
RESNET50 PRE-TRAINED WEIGHTS: THE RESNET50 MODEL, PRE-TRAINED ON IMAGENET, IS USED TO GENERATE EMBEDDINGS—VECTORS REPRESENTING THE ESSENTIAL FEATURES OF THE IMAGES, FACILITATING EASY COMPARISON.
3. SAVING AND LOADING THE DATASET 💾
BACKUP: IT'S ALWAYS GOOD TO HAVE A BACKUP. THE VECTORS ARE SAVED AS A FILE AND CAN BE RELOADED FOR FURTHER PROCESSING IF NEEDED.
4. FINDING SIMILAR CLOTHING 🔍
COSINE DISTANCE: THE VECTOR OF A GIVEN CLOTHING IMAGE IS COMPARED WITH VECTORS IN THE DATASET TO FIND THE MOST SIMILAR CLOTHING ITEMS.
5. K-NEAREST NEIGHBORS 🧑‍🏫
KNN ALGORITHM: THE K-NEAREST NEIGHBORS FOR A GIVEN CLOTHING IMAGE ARE FOUND USING THE NEAREST NEIGHBORS ALGORITHM. THIS ALGORITHM HELPS CLASSIFY OR PREDICT THE CATEGORY OR VALUE OF A NEW SAMPLE BASED ON ITS SIMILARITY TO PREVIOUS SAMPLES.
6. CLUSTERING 📊
KMEANS CLUSTERING: THE CLOTHING IMAGES IN THE DATASET ARE CLUSTERED USING KMEANS, WHICH GROUPS THE SAMPLES INTO SEVERAL CLUSTERS, KEEPING THE INTERNAL PROXIMITY WITHIN EACH CLUSTER.
VISUAL REPRESENTATION: THE CLUSTERS ARE DISPLAYED VISUALLY, ALLOWING YOU TO SEE WHICH IMAGES BELONG TO THE SAME CLUSTER.
INSTALLATION AND USAGE
PREREQUISITES
PYTHON
PYTORCH
NUMPY
MATPLOTLIB
SCIKIT-LEARN
PIL (PILLOW)
HOW TO RUN
INITIALIZE THE MODEL AND DATASET:

python
Copy code
IMGSIM = IMG2VEC(MODEL_NAME='RESNET50', WEIGHTS='IMAGENET')
DATASET_PATH = "/PATH/TO/YOUR/DATASET"
SAVE_PATH = "/PATH/TO/SAVE/EMBEDDINGS"

# EMBEDDING DATASET
IMGSIM.EMBED_DATASET(DATASET_PATH)
IMGSIM.SAVE_DATASET(SAVE_PATH)
LOAD DATASET:

python
Copy code
IMGSIM.LOAD_DATASET(OS.PATH.JOIN(SAVE_PATH, "TENSORS.PT"))
FIND SIMILAR IMAGES:

python
Copy code
TARGET_IMAGE_PATH = "/PATH/TO/YOUR/TARGET_IMAGE.JPG"
SIMILAR_IMAGES = IMGSIM.SIMILAR_IMAGES(TARGET_IMAGE_PATH, N=5)
PRINT("SIMILAR IMAGES:", SIMILAR_IMAGES)
K-NEAREST NEIGHBORS:

python
Copy code
KNN_IMAGES_DETERMINISTIC = IMGSIM.FIND_KNN(TARGET_IMAGE_PATH, K=5, DETERMINISTIC=TRUE)
PRINT("K-NEAREST NEIGHBORS (DETERMINISTIC):", KNN_IMAGES_DETERMINISTIC)
CLUSTERING:

python
Copy code
IMGSIM.CLUSTER_DATASET(NCLUSTERS=6, DISPLAY=TRUE)
CODE STRUCTURE
THE MAIN CLASS, IMG2VEC, HANDLES ALL MAJOR FUNCTIONALITIES:

INITIALIZATION AND MODEL SETUP
DATASET EMBEDDING
FINDING SIMILAR IMAGES
K-NEAREST NEIGHBORS
CLUSTERING AND VISUALIZATION
CONCLUSION
THIS PROJECT ALLOWS YOU TO FIND SIMILAR CLOTHING ITEMS AND CLUSTER THEM USING PRE-TRAINED MODELS AND VARIOUS MACHINE LEARNING TECHNIQUES. IT DEMONSTRATES THE PRACTICAL USE OF DEEP LEARNING MODELS FOR IMAGE PROCESSING AND RETRIEVAL TASKS.

