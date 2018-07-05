import pandas as pd


def load_dataset(path):
    dt = pd.read_csv(path)
   
    images = dt.drop(['label'], axis=1).values
    images = images / 255
    images = images.reshape((images.shape[0],28,28))
    labels = dt['label'].values
    return images, labels
  
   