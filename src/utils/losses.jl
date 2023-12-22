module Losses

export crossentropy
function crossentropy(output, true_values, epsilon=1e-10)
    hstacked_array = hcat(output, true_values)
    return -sum([x_2 * log(x_1 + epsilon) + (1 - x_2) * log(1 - x_1 + epsilon) for (x_1, x_2) in eachrow(hstacked_array)])
end

end