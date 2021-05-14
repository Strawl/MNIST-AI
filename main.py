# This is my first project trying to build an AI
from data.read import Data
import numpy as np
from network import Network


def browse_data(data_object, itype):
    while True:
        x = int(input("please give number: "))
        data_object.display_image(itype, x)


if __name__ == '__main__':
    data = Data("data/")
    data.optimize("train")
    best_network = None
    cost_average = 9999
    for x in range(0,500):
        network = Network([28 * 28, 30, 30, 10])
        feed_forward_temp = np.empty(shape=1000,dtype=np.int16)
        for num in range(0,1000):
            network.feed_forward(data=data, number=num, itype="train")
            feed_forward_temp[num] = network.calculate_cost(data=data, number=num, itype="train")
        t_cost_average = np.average(feed_forward_temp)
        if t_cost_average < cost_average:
            cost_average = t_cost_average
            best_network = network
    print(cost_average)
    print(best_network.correct/1000)

    # browse_data(data_object=data,itype="test")
