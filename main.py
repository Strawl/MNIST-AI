from data.read import *
import numpy as np
from network import *
import logging


# Logger
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler(filename="output.txt")
file_handler.setFormatter(formatter)

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def find_good_network(networks_to_be_created, images_to_be_run, network_shape):
    data = Data("data/")
    best_network = None
    cost_average = 9999
    for x in range(0, networks_to_be_created):
        network = Network(network_shape)
        feed_forward_temp = np.empty(shape=images_to_be_run, dtype=np.double)
        for num in range(0, images_to_be_run):
            output = network.feed_forward(data.get_image(DataSetType.TRAINING,number=num))
            feed_forward_temp[num] = network.calculate_cost(data.get_label(DataSetType.TRAINING,number=num))
            network.flush()
        t_cost_average = np.average(feed_forward_temp)
        if t_cost_average < cost_average:
            cost_average = t_cost_average
            best_network = network
    print(cost_average)
    return network

def train(network: Network, batch_size, batch_count, learning_rate, decay_factor):
    data = Data("data/")
    for i in range(batch_count):
        logger.debug(f"Starting batch number {i}")
        images, labels = data.get_batch(DataSetType.TRAINING, batch_size, random.randrange(60000-batch_size))
        feed_forward_temp = np.empty(shape=batch_size, dtype=np.double)
        for x in range(batch_size):
            network.feed_forward(image=images[x])
            feed_forward_temp[x] = network.calculate_cost(int.from_bytes(labels[x],"big"))
            network.backpropagate()
            network.flush()
        t_cost_average = np.average(feed_forward_temp)
        logger.debug(f"Cost avarage for the batch {i} is {t_cost_average}")
        network.nudge(learning_rate=learning_rate, decay_factor=decay_factor)
        feed_forward_temp = np.empty(shape=batch_size, dtype=np.double)
        for x in range(batch_size):
            network.feed_forward(image=images[x])
            feed_forward_temp[x] = network.calculate_cost(int.from_bytes(labels[x],"big"))
            network.backpropagate()
            network.flush()
        t_cost_average = np.average(feed_forward_temp)
        logger.debug(f"Cost avarage for the batch {i} is {t_cost_average}")
        network.nudge(learning_rate=learning_rate, decay_factor=decay_factor)
        


def run_test_set(network: Network):
    feed_forward_temp = np.empty(shape=10000, dtype=np.double)
    correct = 0
    for num in range(0, 10000):
        output = network.feed_forward(data.get_image(DataSetType.TRAINING,number=num))
        true = data.get_label(DataSetType.TRAINING,number=num)
        np.argmax(output)
        if true == np.argmax(output):
            correct += 1
        feed_forward_temp[num] = network.calculate_cost(data.get_label(DataSetType.TRAINING,number=num))
        network.flush()
    cost_average = np.average(feed_forward_temp)
    logger.debug(f"The average cost is {cost_average}")
    logger.debug(f"Out of 10000 images, {correct} were correct")



if __name__ == '__main__':
    data = Data("data/")
    start = time.time()
    #network = Network([28 * 28, 16, 16, 10])
    network = find_good_network([28 * 28, 16, 16, 10])
    run_test_set(network=network)
    train(network,100,5000,0.01,0.9)
    run_test_set(network=network)
    logger.debug(f"Reading all of this data took {time.time() - start:.2f} seconds") 
