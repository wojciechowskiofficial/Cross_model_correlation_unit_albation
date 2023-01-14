import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from numpy.typing import NDArray
from gc import collect
from joblib import Parallel, delayed
from typing import Union

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
        tf.get_logger().setLevel('INFO')
        self.model_names = model_names
        self.images, self.labels = images, labels
        # load models
        self.base_models = list()
        for m_name in self.model_names:
            self.base_models.append(tf.keras.models.load_model(m_name))
          
    @staticmethod
    def _score_single_channel(preds: list[NDArray[np.float32]], 
                              filter_id: int, 
                              no_channels: int, 
                              images: NDArray[np.float32]) -> float:
        """Part of score_layer method. This method is extracted, 
        because it needs to be static in order to be able to 
        distribute computations.

        :param preds: predictions of itermediate models
        :param filter_id: current filter id to be scored
        :param no_channels: no channels in currently scored layer
        :param images: test images
        :return: a scalar score of a given filter in a given layer
        """            
        per_channel_corr = list()
        for prediction in preds[1:]:
            for channel in range(no_channels):
                first, second = list(), list()
                for i in range(images.shape[0]):
                    # 0th model (the one that is being scored)
                    first.append(np.linalg.norm(preds[0][i,:,:,filter_id]))
                    # 1..nth model
                    second.append(np.linalg.norm(prediction[i,:,:,channel]))
                per_channel_corr.append(pearsonr(first, second)[0])
        return max(per_channel_corr)
            
    def _score_single_layer(self,
                            preds: list[NDArray[np.float32]]) -> list[float]:        
        """Computes convolutional unit scores for a given layer

        :param preds: Avtivation values (layer outputs) of a given layer, 
            for every model
        :return: A list of convolutional unit scores
        """        
        no_channels = preds[0].shape[-1]
        scores = Parallel(n_jobs=-1) \
                (delayed(self._score_single_channel) \
                (preds, filter_id, no_channels, self.images) \
                 for filter_id in range(no_channels))
        return scores

    def score_layers(self, 
                     layers: Union[str, list[str]]) -> dict[str, list[float]]:
        """Score multiple layers, which are given

        :param layers: A layer name or list of layer names to be scored
        :return: A dictionary {layer name : list of scores}
        """  
        if type(layers) == str:
            layers = [layers]      
        scores = dict()
        for layer_name in layers:
            print(f'Currently processing layer {layer_name}', end='\r', flush=True)
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
                preds.append(inter_models[i].predict(self.images, verbose=False))
            # score layer by layer
            scores[layer_name] = self._score_single_layer(preds)
        print()
        return scores
    
    # TODO: change method names and public status
    # TODO: add score all layers method