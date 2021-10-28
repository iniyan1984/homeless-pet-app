import pandas as pd
import numpy as np
import faiss
import cv2
import tensorflow as tf
from api.feature_extractor import FeatureExtractor
import re
import os

local_storage_path = '/app/persistent-folder/data/'
data_path = "/app/api-service/persistent-folder"
DATA_URL = "https://storage.googleapis.com/artifacts.ai5-c1-group1.appspot.com/data/persistent-folder.zip"

MODEL_PATH = local_storage_path + "vision_model"
# load the faiss index
index = faiss.read_index(local_storage_path + 'faiss_index')

# load the feature extractor model
model = tf.keras.models.load_model(MODEL_PATH)
# create an instance of FeatureExtractor class
fe = FeatureExtractor(model)

# load utility dataset
features_df = pd.read_csv(local_storage_path + "features_subset.csv")
features= features_df.set_index('dog_id')
# initialize dogs list
def get_dogs(df, total):
    dogs = []
    for i, row in df.iterrows():
        dog_dict = {}
        dog_dict["id"] = row["AnimalInternal-ID"]
        dog_dict["name"] = row["AnimalName"]
        dog_dict["sex"] = row["AnimalSex"]
        dog_dict["photo_url"] = row["PhotoUrl"]
        colors = re.sub("[\(\)/,]", "", row["AnimalColor"])
        breed = re.sub("[\(\)/,]", "", row["AnimalBreed"])
        tags = colors + " " + breed
        dog_dict["tags"] = tags
        dog_dict["weight"] = row["AnimalCurrentWeightPounds"]
        dog_dict["breed"] = row["AnimalBreed"]
        dog_dict["color"] = row["AnimalColor"]
        dog_dict["age"] = row["Age"]
        dog_dict["memo_text"] = row["MemoText"]
        dogs.append(dog_dict)
    return dogs[:total]

def get_similar_ids(dog_id, k=10):
    # get the feature of query dog
    xq = features.loc[dog_id].astype(np.float32)
    D, I = index.search(np.array([xq]), k)
    # get similar dog ids
    similar_ids = features_df['dog_id'].iloc[I[0]].values
    # return similar ids
    return similar_ids

def get_features(image):
    """ Load and preprocess image."""
    query = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    features = fe.extract_features(query)
    return features

def get_similar_dogs(image, k=10):
    # load image
    xq = get_features(image)
    # get query image id
    # queryID = query_image_path.split(os.path.sep)[1]
    D, I = index.search(np.array([xq]), k)
    # similar dogs ids
    similar_dog_ids = features_df['dog_id'].iloc[I[0]].values
    print("Similar IDs:")
    print(similar_dog_ids)
    return similar_dog_ids

def download_file(packet_url, base_path="", extract=False, headers=None):
    if base_path != "":
        if not os.path.exists(base_path):
            os.mkdir(base_path)
    packet_file = os.path.basename(packet_url)
    with requests.get(packet_url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(os.path.join(base_path, packet_file), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if extract:
        if packet_file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
                zfile.extractall(base_path)
        else:
            packet_name = packet_file.split('.')[0]
            with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                tfile.extractall(base_path)

# # ensure that the data is loaded into the disk
def ensure_data_loaded():
    try:
        print("ensure_data_loaded()")
        if not os.path.exists(data_path):
            print("Downloading...")
            download_file(DATA_URL, base_path=data_path, extract=True)
        else:
            print(data_path, "is available")

        

    except Exception as exc:
        print(str(exc))
