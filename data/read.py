import numpy as np
import logging
import time
from enum import Enum
from PIL import Image

logger = logging.getLogger('root')

# We need to create a singleton, so the data is not read multiple times
class DataSetType(Enum):
    TRAINING = 1
    TESTING = 2
class DataType(Enum):
    LABELS = 1
    IMAGES = 2
class ImageForm(Enum):
    LINEAR = 1
    GRID = 2
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Data(metaclass=Singleton):
    def __init__(self, path: str):
        self.FILES = {
            DataSetType.TRAINING: {
                DataType.LABELS: path + "train-labels.idx1-ubyte",
                DataType.IMAGES: path + "train-images.idx3-ubyte"
            },
            DataSetType.TESTING: {
                DataType.LABELS: path + "train-labels.idx1-ubyte",
                DataType.IMAGES: path + "train-images.idx3-ubyte"
            }
        }
        start = time.time()
        self.DATA = {
            DataSetType.TRAINING: {
                DataType.LABELS: self.load_from_file(self.FILES[DataSetType.TRAINING][DataType.LABELS]),
                DataType.IMAGES: self.load_from_file(self.FILES[DataSetType.TRAINING][DataType.IMAGES])
            },
            DataSetType.TESTING: {
                DataType.LABELS: self.load_from_file(self.FILES[DataSetType.TRAINING][DataType.LABELS]),
                DataType.IMAGES: self.load_from_file(self.FILES[DataSetType.TRAINING][DataType.IMAGES])
            }
        }
        logger.debug(f"Reading all of this data took {time.time() - start:.2f} seconds") 


    def load_from_file(self,file_path: str):
        logger.debug(f"Opening the file : '{file_path}'")
        with open(file_path, "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            logger.debug(f"magic_number determines which type the file is: {magic_number}")
            item_count = int.from_bytes(f.read(4), "big")
            logger.debug(f"There are {item_count} items in this file")
            item_range = range(0,item_count)
            if magic_number == 2051:
                rows = int.from_bytes(f.read(4), "big")
                logger.debug(f"There are {rows} rows per image")
                columns = int.from_bytes(f.read(4), "big")
                logger.debug(f"There are {columns} columns per image")
                item_range = range(0,item_count*rows*columns)
            array = []
            logger.debug(f"Each file is stored in a byte, we are reading bytes one by one")
            for x in item_range:
                array.append(f.read(1))
            
            logger.debug(f"Since traditional arrays don't perform well, we are converting it into a numpy array")
            array = np.array(array)

            if magic_number == 2051:
                array.shape = (item_count, rows * columns)
            return array

    def get_batch(self, data_set_type: DataSetType, amount: int, start: int ):
        images = self.DATA[data_set_type][DataType.IMAGES][start:start+amount]
        labels = self.DATA[data_set_type][DataType.LABELS][start:start+amount]
        return (images,labels)

    def get_label(self, data_set_type: DataSetType, number: int): 
        return int.from_bytes(self.DATA[data_set_type][DataType.LABELS][number],"big")
    
    def get_image(self, data_set_type: DataSetType, number:int, image_form=ImageForm.LINEAR):
        image_raw = np.copy(self.DATA[data_set_type][DataType.IMAGES][number])
        if image_form is ImageForm.GRID:
            image_raw.shape = (28,28)
        return image_raw

    def display_image(self, data_set_type: DataSetType, number: int):
        logger.debug(f"displaying the number {self.get_label(data_set_type,number)}")
        image = Image.fromarray(self.get_image(data_set_type,number,image_form=ImageForm.GRID), mode="L")
        image.show()