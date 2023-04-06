
using .MNISTData, StatsFuns, StatsBase, Flux.Losses


# Activation function for hidden layers
function relu(x)
    return max(0, x)
end

# Derivative of the ReLU activation function
function relu_prime(x)
    return x > 0 ? 1 : 0
end


# Network structure
mutable struct Network
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    topology::Vector{Int64}
    network_size::Int64
end

# Function to create a new neural network with a given topology
function create_network(topology)
    weights = Matrix{Float64}[]
    biases = Vector{Float64}[]
    for i in 2:length(topology)
        push!(weights, randn(topology[i-1], topology[i]) .* sqrt(2 / (topology[i-1] + topology[i])))
        push!(biases, zeros(topology[i]))
    end
    return Network(weights, biases, topology, length(topology))
end

# Function to perform a feed-forward pass through the network, given an input activation
function feed_forward(network::Network, activation)
    activations = Vector{Float64}[]
    push!(activations, activation)
    for i in 1:network.network_size-2
        activation = permutedims(network.weights[i]) * activation .+ network.biases[i]
        push!(activations, relu.(activation))
    end
    activation = permutedims(network.weights[end]) * activation .+ network.biases[end]
    push!(activations, activation)
    return activations
end

# Function to test the performance of the neural network on the test dataset
function test(network::Network)
    cost = 0
    correct = 0
    for (num, label, image) in get_test_batch(1, 10000, true)
        output = StatsFuns.softmax(feed_forward(network, image)[end])
        true_values = zeros(10)
        true_values[label+1] = 1
        cost += Losses.crossentropy(output, true_values)
        if argmax(output) - 1 == label 
            correct += 1
        end
    end
    println("The network identified ", correct, " images correctly out of 10000")
    println("The netowork came out with an average cost of ", cost / 10000)
end


#Function to train the neural network using stochastic gradient descent and momentum
function train(network::Network, iterations, batch_size, learning_rate, decay_factor)
    previous_weight_gradients = Matrix{Float64}[]
    previous_bias_gradients = Vector{Float64}[]
    for i in 2:length(network.topology)
        push!(previous_weight_gradients, zeros(network.topology[i-1], network.topology[i]))
        push!(previous_bias_gradients, zeros(network.topology[i]))
    end
    for i in 1:iterations
        batch = get_train_batch(batch_size)
        total_cost = 0
        total_correct = 0

        bias_gradients = Matrix{Float64}[]
        weight_gradients = Array{Float64,3}[]
        for l in 1:network.network_size-1
            push!(bias_gradients, zeros(Float64, batch_size, network.topology[l+1]))
            push!(weight_gradients, zeros(Float64, network.topology[l], network.topology[l+1], batch_size))
        end
        for (j, (label, image)) in enumerate(batch)
            activations = feed_forward(network, image)
            output = StatsFuns.softmax(activations[end])
            if argmax(output) - 1 == label
                total_correct += 1
            end
            true_values = zeros(10)
            true_values[label+1] = 1
            total_cost += Losses.crossentropy(output, true_values)
            z = output .- true_values
            for l in reverse(1:network.network_size-1)
                bias_gradients[l][j, :] = z
                for (n, z_n) in enumerate(z)
                    weight_gradients[l][:, n, j] = activations[l] * z_n
                end
                if l != 1
                    activation_gradients = mean(permutedims(network.weights[l]) .* vec(z), dims=1)
                    z = relu_prime.(activation_gradients)
                end
            end
        end
        for l in 1:network.network_size-1
            previous_bias_gradients[l] = (1 - decay_factor) * vec(learning_rate * permutedims(mean(bias_gradients[l], dims=1))) + decay_factor * previous_bias_gradients[l]
            previous_weight_gradients[l] = (1 - decay_factor) * (learning_rate * reshape(mean(weight_gradients[l], dims=3), (network.topology[l], network.topology[l+1]))) + decay_factor * previous_weight_gradients[l]
            network.biases[l] = network.biases[l] .- previous_bias_gradients[l]
            network.weights[l] = network.weights[l] .- previous_weight_gradients[l]
        end
        avg_cost = total_cost / batch_size
        accuracy = total_correct / batch_size
        println("iteration $i, cost: $(round(avg_cost, digits=3)), accuracy: $(round(accuracy, digits=3))")
    end
end

