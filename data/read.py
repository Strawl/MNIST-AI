import numpy as np
from PIL import Image


class Data:

    def __init__(self, path):
        self.TRAIN_FILES = [
            path + "train-labels.idx1-ubyte",
            path + "train-images.idx3-ubyte"
        ]
        self.TEST_FILES = [
            path + "t10k-labels.idx1-ubyte",
            path + "t10k-images.idx3-ubyte"
        ]
        self.data_array = [None] * 4

    def read_data(self, itype, files=None):
        if itype == "train":
            files = self.TRAIN_FILES
        elif itype == "test":
            files = self.TEST_FILES
        else:
            return
        with open(files[0], "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            item_count = int.from_bytes(f.read(4), "big")
            array = []
            for items in range(0, item_count):
                array.append(f.read(1))
            array = np.array(array)
            if itype == "train":
                self.data_array[0] = array
            elif itype == "test":
                self.data_array[2] = array

        with open(files[1], "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            image_count = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            columns = int.from_bytes(f.read(4), "big")
            array = []
            for items in range(0, image_count * rows * columns):
                array.append(f.read(1))
            array = np.array(array)
            array.shape = (image_count, rows, columns)
            if itype == "train":
                self.data_array[1] = array
            elif itype == "test":
                self.data_array[3] = array

    def display_image(self, itype, number):
        if itype == "train":
            if self.data_array[0] is None:
                self.read_data(itype)
            print(int.from_bytes(self.data_array[0][number], "big"))
            image = Image.fromarray(self.data_array[1][number], mode="L")
            image.show()
        elif itype == "test":
            if self.data_array[2] is None:
                self.read_data(itype)
            print(int.from_bytes(self.data_array[2][number], "big"))
            image = Image.fromarray(self.data_array[3][number], mode="L")
            image.show()
            pass
