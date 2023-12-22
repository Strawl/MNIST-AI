const WORKSPACE="/home/dominik/Documents/ai-mnist-learn/MNIST-AI"
const DATA_DIR = WORKSPACE * "/data"
const SAVED_NETWORKS_DIR = WORKSPACE * "/saved_networks"


if !isdir(DATA_DIR)
    mkdir(DATA_DIR)
end


if !isdir(SAVED_NETWORKS_DIR)
    mkdir(SAVED_NETWORKS_DIR)
end