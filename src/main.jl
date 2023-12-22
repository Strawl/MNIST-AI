include("utils/data_loader.jl")
include("utils/losses.jl")
include("utils/activation_functions.jl")
include("utils/network_loader.jl")
include("networks/fnn.jl")
include("training/train_fnn.jl")
include("settings.jl")
using .MNISTLoader
using .NetworkLoader
using .ActivationFunctions
using .Losses
using .FNN
using .TrainFNN


function format_number(number, precision)
    factor = 10^precision
    formatted = round(number * factor) / factor
    return formatted
end

function guess(network::FNN.Network)
    for (num, label, image) in MNISTLoader.get_test_batch(1, 100)
        output = ActivationFunctions.softmax(FNN.feed_forward(network, image)[end])
        println(format_number.(output,3))
        MNISTLoader.display_image(num,true)
        println("AI: The image above is: $(argmax(output) - 1)")
        readline()
    end
end

function play_guess(network_hash::String)
    network = NetworkLoader.load_network(network_hash)
    println("Now let's play a guessing game")
    guess(network)
end

function main()
    network = FNN.create_network([784,10])
    println("Testing the network before training:")
    TrainFNN.test(network)
    
    TrainFNN.train(network, 3, 300, 0.001, 0.9)
    filename = NetworkLoader.hash_network(network)
    NetworkLoader.save_network(network, "$SAVED_NETWORKS_DIR/$filename")

    println("Testing the network after training:")
    TrainFNN.test(network)
end

