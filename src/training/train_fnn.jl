module TrainFNN

using ..FNN, ..MNISTLoader, ..ActivationFunctions, ..Losses
using ProgressMeter, Statistics

export test, train
# Function to test the performance of the neural network on the test dataset
function test(network::FNN.Network)
    cost = 0
    amount_incorrect = zeros(10)
    amount_correct = zeros(10)
    for (num, label, image) in MNISTLoader.get_test_batch(1, 10000)
        output = ActivationFunctions.softmax(feed_forward(network, image)[end])
        true_values = zeros(10)
        true_values[label+1] = 1
        cost += Losses.crossentropy(output, true_values)
        if argmax(output) - 1 == label 
            amount_correct[label+1] += 1
        else
            amount_incorrect[label+1] += 1
        end
    end
    println("The network identified $(sum(amount_correct)) images correctly out of 10000")
    println("The netowork came out with an average cost of $(cost / 10000)")
    for (i, (correct, incorrect)) in enumerate(zip(amount_correct, amount_incorrect))
        println("The number $(i-1) was guessed $(correct) times correctly and $(incorrect) times incorrectly")
    end

end


#Function to train the neural network using stochastic gradient descent and momentum
function train(network::FNN.Network, epochs, batch_size, learning_rate, decay_factor)
    previous_weight_gradients = Matrix{Float64}[]
    previous_bias_gradients = Vector{Float64}[]
    for i in 2:length(network.topology)
        push!(previous_weight_gradients, zeros(network.topology[i-1], network.topology[i]))
        push!(previous_bias_gradients, zeros(network.topology[i]))
    end
    for epoch in 1:epochs
        println("Starting the epoch number $epoch")
        batches = get_batches(batch_size)
        num_batches = length(batches)
        progress = Progress(num_batches, dt=0.0, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        generate_showvalues(batch_idx, avg_cost, accuracy) = () -> [(:Batch, batch_idx), (:Avg_Cost, round(avg_cost, digits=3)), (:Accuracy, round(accuracy, digits=3))]
        total_cost = 0
        total_correct = 0
        for (batch_idx, batch) in enumerate(batches)
            cost = 0
            correct = 0
            bias_gradients = Matrix{Float64}[]
            weight_gradients = Array{Float64,3}[]
            for l in 1:network.network_size-1
                push!(bias_gradients, zeros(Float64, batch_size, network.topology[l+1]))
                push!(weight_gradients, zeros(Float64, network.topology[l], network.topology[l+1], batch_size))
            end
            Threads.@threads for (j, (label, image)) in collect(enumerate(batch))
                activations = feed_forward(network, image)
                output = ActivationFunctions.softmax(activations[end])
                if argmax(output) - 1 == label
                    correct += 1
                end
                true_values = zeros(10)
                true_values[label+1] = 1
                cost += Losses.crossentropy(output, true_values)
                z = output .- true_values
                for l in reverse(1:network.network_size-1)
                    bias_gradients[l][j, :] = z
                    for (n, z_n) in enumerate(z)
                        weight_gradients[l][:, n, j] = activations[l] * z_n
                    end
                    if l != 1
                        z = (network.weights[l] * z) .* ActivationFunctions.leakyrelu_prime.(activations[l])
                    end
                end
            end
            Threads.@threads for l in 1:network.network_size-1
                previous_bias_gradients[l] = (1 - decay_factor) * vec(learning_rate * permutedims(mean(bias_gradients[l], dims=1))) + decay_factor * previous_bias_gradients[l]
                previous_weight_gradients[l] = (1 - decay_factor) * (learning_rate * reshape(mean(weight_gradients[l], dims=3), (network.topology[l], network.topology[l+1]))) + decay_factor * previous_weight_gradients[l]
                network.biases[l] = network.biases[l] .- previous_bias_gradients[l]
                network.weights[l] = network.weights[l] .- previous_weight_gradients[l]
            end
            avg_cost = cost / batch_size
            accuracy = correct / batch_size
            total_correct += correct
            total_cost += cost
            #println("iteration $i, cost: $(round(avg_cost, digits=3)), accuracy: $(round(accuracy, digits=3))")
            ProgressMeter.next!(progress, showvalues = generate_showvalues(batch_idx, avg_cost, accuracy))
        end
        avg_cost = total_cost / batch_size * num_batches 
        accuracy = total_correct / batch_size * num_batches
        ProgressMeter.finish!(progress, showvalues = generate_showvalues(num_batches, avg_cost, accuracy))
        test(network)
    end
end

end