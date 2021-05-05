from typing import Callable

import tensorflow as tf


class DataReader:
    """Abstraction of the data reader for efficient train-test split.
    """
    def dataset(self) -> tf.data.Dataset:
        """Return file reader.
        Returns:
            file-format datum reader.
        """
        raise NotImplementedError('DataReader.rawset is not implemented')
    
    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor, required format 
                text: tf.string, text.
                speech: [tf.float32; T], speech signal in range (-1, 1).
        """
        raise NotImplementedError('DataReader.preproc is not implemented')
