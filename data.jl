# Automatically generated file, do not edit!

include("./utilities.jl")
using Glob, Random

loadimgpaths(path::String) = glob("*/*/*.dcm", path)
path2id(imgpath::String) = match(r"(?<=\/).[^\/]+(?=.dcm)", imgpath).match
id2mask(id, csv) = csv[findfirst(x->x==id, csv[:, 1]), 2]

function batch(imgpaths, csv; imsize=nothing)
    imgs = opendcm.(imgpaths)
    
    masks = rle2mask.(id2mask.(path2id.(imgpaths), [csv]), [size(imgs[1])])
    if !isnothing(imsize)
        imgs = imresize.(imgs, [imsize])
        masks = imresize.(masks, [imsize])
    end
    return Float64.(cat(imgs..., dims=4)), Bool.(ceil.(cat(masks..., dims=4)))
end

dataloader(imgpaths, csv, bs; imsize=nothing) = (batch(imgpaths, csv, imsize=imsize) for imgpaths in Iterators.partition(imgpaths, bs))
