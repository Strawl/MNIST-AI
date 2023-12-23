module FNN

export FNN, create_network, feed_forward, Network
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
        # For ReLU
        #push!(weights, randn(topology[i-1], topology[i]) .* sqrt(2 / (topology[i-1] + topology[i])))
        push!(weights, randn(topology[i-1], topology[i]) .* sqrt(2 / ((1 + 0.01^2) * (topology[i-1] + topology[i]))))

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
        push!(activations, ActivationFunctions.leakyrelu.(activation))
    end
    activation = permutedims(network.weights[end]) * activation .+ network.biases[end]
    push!(activations, activation)
    return activations
end

end