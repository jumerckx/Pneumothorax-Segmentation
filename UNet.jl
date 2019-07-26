include("./data.jl")
using Flux, CuArrays
using Flux: @treelike

imgpaths = loadimgpaths("./data/train-test/dicom-images-train")
csv = CSV.read("./data/train-test/train-rle.csv")

loader = dataloader(imgpaths[1:200], csv, 2, imsize=(256, 256))

# function centercrop(x, target_size)
#     start_size = ((size(x)[[1, 2]] .- target_size) .÷ 2 .+ 1)
#     x[start_size[1]:(start_size[1] + target_size[1] - 1), start_size[2]:(start_size[2] + target_size[2] - 1), :, :]
# end

using Flux, CuArrays
using Flux: @treelike

Flux.@adjoint function cat(A::AbstractArray...; dims::Int)
    sz = cumsum([size.(A, dims)...])
    return cat(A...; dims=dims), Δ->(map(n->Δ[fill(Colon(), dims - 1)..., sz[n]-size(A[n], dims)+1:sz[n], fill(Colon(), ndims(A[n]) - dims)...], eachindex(A))...,)
end

ConvBlock(ch::Pair{<:Integer,<:Integer}, σ = relu; pad=1, kwargs...) = Chain(
    Conv((3,3), ch, identity, pad=pad, kwargs...), Conv((3,3), ch[2]=>ch[2], identity, pad=pad, kwargs...)
)

ConvUpBlock(ch::Pair{<:Integer,<:Integer}, σ=relu; pad=1, kwargs...) = Chain(
    ConvTranspose((2,2), ch, stride=(2,2)),
    ConvBlock(ch, σ; pad=pad, kwargs...)...
)

contraction() = Chain(
    ConvBlock(1=>16, relu),
    ConvBlock(16=>32, relu),
    ConvBlock(32=>64, relu),
    ConvBlock(64=>128, relu)
)

expansion() = Chain(
    ConvUpBlock(256=>128, relu),
    ConvUpBlock(128=>64, relu),
    ConvUpBlock(64=>32, relu),
    ConvUpBlock(32=>16, relu)
)

struct UNet
    contraction
    pool
    connector
    expansion
    out
end
@treelike UNet

UNet() = UNet(
    contraction(),
    MaxPool((2,2)),
    ConvBlock(128=>256, relu),
    expansion(),
    Conv((1, 1), 16=>1, sigmoid)
)

function (m::UNet)(x)
    intermediary = []
    for (i, block) in enumerate(m.contraction)
        x = block(x)
        println("1")
        intermediary = [intermediary; [x]]
        println("2")
        x = m.pool(x)
    end
    x = m.connector(x)
    for (i, block) in enumerate(m.expansion)
        x = block[1](x)
        x = cat(x, intermediary[5-i], dims=3)
        x = block[2:3](x)
    end
    x = m.out(x)
    return(x)
end

model = UNet()|>gpu

input = first(loader)[1]|>gpu

@time model(input)


@time Flux.gradient(model) do model
    sum(model(input))
end
