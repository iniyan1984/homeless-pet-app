import pandas as pd
import numpy as np
import faiss
import cv2
import tensorflow as tf
from api.feature_extractor import FeatureExtractor
import re
import os

local_storage_path = '/app/persistent-folder/data/'
data_path = "persistent-folder"

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

# def download_from_gdrive(file_id, file_name):
#     os.system("wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget \
#     --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
#     'https://docs.google.com/uc?export=download&id=$file_id' -O- | \
#     sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id\" \
#     -O $file_name && rm -rf /tmp/cookies.txt")

#     print("{} downloaded!".format(file_name))

# # ensure that the data is loaded into the disk
# def ensure_data_loaded():
#     try:
#         print("ensure_data_loaded()")
#         if not os.path.exists(data_path):
#             print("Downloading...")
#             download_from_gdrive("1qiacanLXiUndgAjOwFSC2r7WGPUcqR0r", "persistent-folder.zip")
#         else:
#             print(data_path, "is available")

        

#     except Exception as exc:
#         print(str(exc))
