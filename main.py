# This is my first project trying to build an AI
from data.read import Data
from PIL import Image
from network import Network


def browse_data(data_object, itype):
    while True:
        x = int(input("please give number: "))
        data_object.display_image(itype, x)


if __name__ == '__main__':
    network = Network([28 * 28, 30, 30, 10])
    data = Data("data/")
    data.optimize("train")
    network.feed_forward(data=data, number=15)

    # browse_data(data_object=data,itype="test")
