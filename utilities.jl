# Automatically generated file, do not edit!

using JSON, DelimitedFiles, Images, DICOM, ImageMagick, CSV
export notebook2script, visualize

function notebook2script(nb)
    url = replace(nb, ".ipynb"=>".jl")
    name = titlecase(match(r"(?<=\/).[^\/]+(?=.ipynb)", nb).match)
    script = []
    open(nb) do file
        for cell in JSON.parse(file)["cells"]
            if length(cell["source"])>0 && cell["source"][1] == "#export\n"
                push!(script, "", strip.(cell["source"][2:end], '\n'))
            end
        end
    end
    template = "# Automatically generated file, do not edit!"
    script = [template; script...]
    writedlm(url, script, quotes=false)
end

function opendcm(filename)
    filename|> DICOM.dcm_parse |> x->(x)[tag"Pixel Data"] |> ImageMagick.load_
end

function rle2mask(rle, size)
    mask = Bool.(zeros(Int, size))
    rle = parse.(Int, split(rle))
    if rle[1] == -1 return mask end
    count = 0
    for i in 1:2:length(rle)
        count += rle[i]
        mask[count:count+rle[i+1]] .= 1
        count += rle[i+1]
    end
    return mask
end
    
function mask2rle(mask, string=true)
    rle = []
    count = 0
    while true
        masked = findnext(mask, count+1)
        if isnothing(masked) break end
        
        unmasked = findnext(.!mask, masked)
        if isnothing(unmasked) unmasked = length(mask) end
            
        push!(rle, masked-count, unmasked-1-masked)
        count = unmasked-1
    end
    if isempty(rle) rle = [-1] end
    if string rle = join(rle, " ") end
    return rle
end

visualize(img::I, mask::M) where {I, M <: AbstractArray} = colorview(RGB, img, img.+0.2.*mask, img.+0.4.*mask)

function visualize(imgid::T, rlemask::T; datapath="D:/code/kaggle/data/samples"::String) where {T <: String}
    img = opendcm(datapath*"/"*imgid*".dcm")
    mask = rle2mask(rlemask, size(img))
    visualize(img, mask)
end

function visualize(imgid::T, rlemask::T, id2path::Dict) where {T <: String}
    img = opendcm(id2path[imgid])
    mask = rle2mask(rlemask, size(img))
    visualize(img, mask)
end
