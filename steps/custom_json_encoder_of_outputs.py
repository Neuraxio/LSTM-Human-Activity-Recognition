from neuraxle.rest.flask import JSONDataResponseEncoder
import numpy as np


class CustomJSONEncoderOfOutputs(JSONDataResponseEncoder):
    """This is a custom JSON response encoder class for converting the pipeline's transformation outputs."""

    def encode(self, data_inputs) -> dict:
        """
        Convert predictions to a dict for creating a JSON Response object.

        :param data_inputs:
        :return:
        """
        return {
            'predictions': list(np.array(data_inputs).tolist())
        }
