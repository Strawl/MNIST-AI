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
        self.optimized_data_array = [None] * 4

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

    def get_image(self, itype, number, opt):
        if opt:
            if itype == "train":
                if self.optimized_data_array[0] is None:
                    self.optimize(itype)
                return self.optimized_data_array[1][number]
            elif itype == "test":
                if self.optimized_data_array[2] is None:
                    self.optimize(itype)
                return self.optimized_data_array[3][number]
        else:
            if itype == "train":
                if self.data_array[0] is None:
                    self.read_data(itype)
                return self.data_array[1][number]
            elif itype == "test":
                if self.data_array[2] is None:
                    self.read_data(itype)
                return self.data_array[3][number]

    def get_label(self, itype, number, opt):
        if opt:
            if itype == "train":
                if self.optimized_data_array[0] is None:
                    self.optimize(itype)
                return int.from_bytes(self.optimized_data_array[0][number], "big")
            elif itype == "test":
                if self.optimized_data_array[2] is None:
                    self.optimize(itype)
                return int.from_bytes(self.optimized_data_array[2][number], "big")
        else:
            if itype == "train":
                if self.data_array[0] is None:
                    self.read_data(itype)
                return int.from_bytes(self.data_array[0][number], "big")
            elif itype == "test":
                if self.data_array[2] is None:
                    self.read_data(itype)
                return int.from_bytes(self.data_array[2][number], "big")

    def display_image(self, itype, number):
        print(self.get_label(itype, number,False))
        image = Image.fromarray(self.get_image(itype, number,False), mode="L")
        image.show()

    def optimize(self,itype):
        if itype == "train":
            if self.data_array[0] is None:
                self.read_data(itype)
            self.optimized_data_array[0] = self.data_array[0]
            self.optimized_data_array[1] = np.copy(self.data_array[1])
            self.optimized_data_array[1].shape = (self.optimized_data_array[1].shape[0],
                                                  (self.optimized_data_array[1].shape[1] *
                                                   self.optimized_data_array[1].shape[2]))
        elif itype == "test":
            if self.data_array[2] is None:
                self.read_data(itype)
            self.optimized_data_array[2] = self.data_array[2]
            self.optimized_data_array[3] = np.copy(self.data_array[3])
            self.optimized_data_array[3].shape = (self.optimized_data_array[3].shape[0],
                                                  (self.optimized_data_array[3].shape[1] *
                                                   self.optimized_data_array[3].shape[2]))

