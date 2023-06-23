#!/usr/bin/env python3
import numpy as np
from sklearn.externals import joblib     # 直接用import joblib也一样


class NeuronalFem:
    """


    Args:
        param1(str): Local path directory to machine learning model and
                    scalers.

    Attributes:
        estimator(pickle obj): Estimator ready for prediction using the
                              multi−layer perceptron model (ANN).
        scaler_x(pickle obj): Contains the binary file, which has the methods
                              to be used for scaling features.
        scaler_y(pickle obj): Contains the binary file, which has the methods
                              to be used for scaling predictions.

    """

    def __init__(self, modeldir):

        self.estimator = joblib.load(modeldir + 'ann.pkl')
        self.scalerx = joblib.load(modeldir + 'scaler_x.pkl')
        self.scalery = joblib.load(modeldir + 'scaler_y.pkl')

    def get_features(self, filepath):
        """

        Args :
            param1 (str): Local path directory to the file containing
                          features
            Returns :
                Array containing the features (total stress tensor,
                viscoplastic strain tensor, back stress tensor,drag stress
                and plastic strain.)
        """
        file = open(filepath, 'r')
        for line in file.readlines():
            features = line.rstrip().split(',')
        features = [float(i) for i in features]
        file.close()
        return [features]

    def save_predictions(self, output, filepath):
        """

        Args :
            param1 (np.array): Array with the output predictions given
                                by estimator(ANN).
            param2 (str): Local path directory to the file where
                            predictions are stored
        Returns :
            None.
        """
        file = open(filepath, 'wb')
        # Initialize predictions vector
        predictions = np.arange(8, dtype=float)
        for i in range(0,8):
            predictions[i]=output[0][i]
        np.savetxt(file, [predictions], fmt='%0.6f', delimiter=',')
        file.close()
        return None

# Main program


if __name__ == "__main__":
    # Machine learning model directory
    modeldir = '/home/miguel/Documents/tese/ViscoPlastic−ML/2D/train/model/'
    # Initialize neuronalfem class with trained model for further prediction
    # and scalers to transform the data.
    neuronalfem = NeuronalFem(modeldir)
    # Load features from features.txt file
    features = neuronalfem.get_features('/home/miguel/features.txt')
    # Transform features values to make predictions
    input = neuronalfem.scaler_x.transform(features)
    # Make predictions and transform the output
    output = neuronalfem.scaler_y.inverse_transform(
                (neuronalfem.estimator.predict(input)))
    # Save predictions
    neuronalfem.save_predictions(output, '/home/miguel/predictions.txt')
