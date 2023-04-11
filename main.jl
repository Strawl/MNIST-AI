include("mnist_data.jl")
using .MNISTData
include("neural_network.jl")

# This function takes a trained neural network and uses it to make predictions
# for images from the test dataset. It displays the images and prints the
# predicted label for each.

function format_number(number, precision)
    factor = 10^precision
    formatted = round(number * factor) / factor
    return formatted
end

function guess(network::Network)
    for (num, label, image) in MNISTData.get_test_batch(1, 100)
        output = StatsFuns.softmax(feed_forward(network, image)[end])
        println(format_number.(output,3))
        display_image(num,true)
        println("AI: The image above is: $(argmax(output) - 1)")
        readline()
    end
end

# The main function initializes the neural network, tests its performance
# before and after training, and then plays a guessing game with the user.
function do_training()
    network = create_network([784, 512,  256, 128, 10])
    println("Testing the network before training:")
    test(network)
    
    train(network, 30, 500, 0.001, 0.9)
    filename = hash_network(network)
    save_network(network, "./networks/$filename")

    println("Testing the network after training:")
    test(network)

end

function play_guess(network_hash::String)
    network = load_network(network_hash)
    println("Now let's play a guessing game")
    guess(network)
end

function main()
   do_training() 
end

