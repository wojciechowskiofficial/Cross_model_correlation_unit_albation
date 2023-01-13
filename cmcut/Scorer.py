import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from numpy.typing import NDArray
from gc import collect

class Scorer:  
    """Used for scoring the convolutional units
    """    
    def __init__(self, 
                 model_names: list[str], 
                 images: NDArray[np.float32], 
                 labels: NDArray[np.uint8]) -> None:     
        """Loads models and test data

        :param model_names: A list of paths pointing to .h5 format models
        :param images: Test images
        :param labels: Test labels
        """        
        self.model_names = model_names
        self.images, self.labels = images, labels
        # load models
        self.base_models = list()
        for m_name in self.model_names:
            self.base_models.append(tf.keras.models.load_model(m_name))
            
    def score_layer(self,
                    preds: list[NDArray[np.float32]]) -> list[float]:        
        """Computes convolutional unit scores for a given layer

        :param preds: Avtivation values (layer outputs) of a given layer, 
            for every model
        :return: A list of convolutional unit scores
        """        
        scores = list()
        no_channels = preds[0].shape[-1]
        for score in range(no_channels):
            print(score, end='\r', flush=True)
            per_channel_corr = list()
            for prediction in preds[1:]:
                for channel in range(no_channels):
                    first, second = list(), list()
                    for i in range(self.images.shape[0]):
                        # 0th model (the one that is being scored)
                        first.append(np.linalg.norm(preds[0][i,:,:,score]))
                        # 1..nth model
                        second.append(np.linalg.norm(prediction[i,:,:,channel]))
                    per_channel_corr.append(pearsonr(first, second)[0])
            scores.append(max(per_channel_corr))
        return scores

    def score_multiple_layers(self, 
                              layers: list[str]) -> dict[str, list[float]]:
        """Score multiple layers, which are given

        :param layers: A list of layer names to be scored
        :return: A dictionary {layer name : list of scores}
        """        
        scores = dict()
        for layer_name in layers:
            # create intermediate models
            inter_models = list()
            collect()
            for i in range(len(self.model_names)):
                input_layer = self.base_models[i].input
                output_layer = self.base_models[i].get_layer(layer_name).output
                inter_models.append(tf.keras.Model(inputs=input_layer, 
                                                   outputs=output_layer))
            # compute itermediate predictions
            preds = list()
            collect()
            for i in range(len(self.model_names)):
                preds.append(inter_models[i].predict(self.images))
            scores[layer_name] = self.score_layer(preds)
        return scores