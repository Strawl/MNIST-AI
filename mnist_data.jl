module MNISTData
using HTTP, Downloads, ImageInTerminal, Random, Images, CodecZlib, Statistics


export test_images, train_images, test_labels, train_labels, get_test_batch, get_batches, display_image

data_dir = "./data"
urls_filenames = [
    ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train_labels"),
    ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train_images"),
    ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "test_images"),
    ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "test_labels")
]

if !isdir(data_dir)
    mkdir(data_dir)
end


for (url, filename) in urls_filenames
    filepath = joinpath(data_dir, filename * ".gz")
    decompressed_filepath = joinpath(data_dir, filename)
    if !isfile(decompressed_filepath)
        if !isfile(filepath)
            println("Downloading $filename from $url")
            Downloads.download(url, filepath)
        else
            println("$filename already exists. Skipping download.")
        end
        println("Decompressing $filename")
        open(decompressed_filepath, "w") do output
            stream = GzipDecompressorStream(open(filepath))
            write(output, stream)
            close(stream)
        end
    else
        println("$filename is already decompressed. Skipping.")
    end
end


train_labels = permutedims(read("./data/train_labels")[9:60000+8])
test_labels = permutedims(read("./data/test_labels")[9:10000+8])
test_images = permutedims(reshape(read("./data/test_images")[17:10000*784+16],(784,10000)))
train_images = permutedims(reshape(read("./data/train_images")[17:60000*784+16],(784,60000)))


function get_batches(batch_size)
    epoch = collect(zip(Int.(train_labels), eachrow(normalize_data(train_images))))
    shuffle!(epoch)

    # Calculate the number of batches
    num_batches = ceil(Int, length(epoch) / batch_size)

    # Divide the shuffled data into batches
    batches = [epoch[(i - 1) * batch_size + 1:min(i * batch_size, end)] for i in 1:num_batches]

    return batches
end


function get_test_batch(start, amount, test::Bool=false)
    return collect(zip(collect(start:start+amount-1), Int.(test_labels[start:start+amount-1]), eachrow(normalize_data(test_images[start:start+amount-1,:]))))
end



function display_image(num::Int64,test::Bool=false)
    if !test
        img = colorview(Gray,normedview(permutedims(reshape(MNISTData.train_images[num,:],(28,28)))))
    else
        img = colorview(Gray,normedview(permutedims(reshape(MNISTData.test_images[num,:],(28,28)))))
    end
    display(img)
end


function normalize_data(data)
    mean_data = mean(data)
    std_data = std(data)
    return (data .- mean_data) ./ std_data
end

end