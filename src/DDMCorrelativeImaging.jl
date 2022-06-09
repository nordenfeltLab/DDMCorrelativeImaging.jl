module DDMCorrelativeImaging

#packages
using DDMFramework
using BioformatsLoader
using RegionProps
using Images
using SegmentationUtils
using SparseArrays

using JSON
using DataFrames
using URIs
using HTTP

using LinearAlgebra, Distances, CoordinateTransformations
using Statistics
using StatsBase


export CorrelativeState

include("triangle_features.jl")
include("comparison.jl")
include("segmentation.jl")

struct Matched
    hit_rate::Float64
    points::Matrix{Float64}
    matched_points::Matrix{Float64}
end


struct CorrelativeState
    matched::Vector{Matched}
    comparison_data::Vector{Matrix{Float64}}
    responses::Vector{Int64}
    config::Dict{String,Any}
end

struct FOV
    height::Int64
    width::Int64
    pixelmicrons::Float64
    stage_y::Float64
    stage_x::Float64
end


function query_ddm(params::Dict{String,Any})
     plugin, exp_id, port, filter = let p = params["analysis"]
        port = haskey(p, "port") ? p["port"] : 4445
        filter = haskey(p, "dia_filter") ? p["dia_filter"] : """filter:{timepoint:[{op:"max",args:[]}]}"""
        p["plugin"],p["exp_id"],port,filter
    end
    query = "{$plugin($filter){fov_id last_stage_pos}}"
    println("query experiment $exp_id ($plugin) on port $port with $query")
    url = "http://localhost:$port/api/v1/experiments/$exp_id/?query=$(escapeuri(query))"
    response = HTTP.request(:GET, url)
    let data = JSON.parse(String(response.body))
        d = data[plugin]
        map(unique(d["fov_id"])) do id
            idx = d["fov_id"] .==id
            mapreduce(vcat,d["last_stage_pos"][idx]) do x
                Float64.([x[2] x[1]])
            end
        end
    end
end
    

function CorrelativeState(params::Dict{String,Any})
    comparison_data = query_ddm(params)
    CorrelativeState(
        Vector{Matched}[],
        comparison_data,
        Vector{Float64}[],
        params
        )
end


function parse_image_meta!(state,data)
    
    function push_new_lazy!(fun,d,k)
        if !haskey(d,k)
            push!(d,k => fun())
        end
        return d
    end
    
    push_new_lazy!(state.config, "image_meta") do
        Dict{String, Any}(
            "stage_pos_x" => [],
            "stage_pos_y" => []
        )
    end
    
    let (mdata, params) = (data["image"].Pixels, state.config["image_meta"])
        push!(params["stage_pos_x"], mdata[:Plane][1][:PositionX])
        push!(params["stage_pos_y"], mdata[:Plane][1][:PositionY])
        
        push_new_lazy!(params, "img_size") do
            (y=mdata[:SizeY],x=mdata[:SizeX],z=mdata[:SizeZ])
        end
        push_new_lazy!(params, "pixelmicrons") do
            mdata[:PhysicalSizeX]
        end
    end
end

function drop_empty_dims(img::ImageMeta)
    dims = Tuple(findall(x -> x.val.stop == 1, img.data.axes))
    dropdims(img.data.data, dims=dims)
end
function DDMFramework.handle_update(state::CorrelativeState, data)
    parse_image_meta!(state,data)
    update_state!(
        state,
        drop_empty_dims(data["image"])
    )
end

to_named_tuple(dict::Dict{K,V}) where {K,V} = NamedTuple{Tuple(Iterators.map(Symbol,keys(dict))), NTuple{length(dict),V}}(values(dict))
function update_state!(state::CorrelativeState, image)
    
    seg_p = state.config["segmentation"]
    fun_p = seg_p["function"]
    ref_c = seg_p["primary_channel"]
    df = let img = image[ref_c,:,:]
        segmentation_handler(
            img;
            to_named_tuple(fun_p)...
        )
    end
    stage_coords = to_stage_coords(
        df,
        state.config["system"]["camera_M"],
        state.config["image_meta"]
    )
    matches = get_matches(stage_coords, state.comparison_data)
    display("hitrate: $(matches[1])")
    
    response = if matches[1] > 1/3
        push!(state, Matched(matches...))
        push!(state.responses, 1)
        "1"
    else
        push!(state.responses, 0)
        "0"
    end
    
    (response, state)
end

function to_stage_coords(camera_m, x,y, fov::FOV)
    translation(x,p) = x .* p.pixelmicrons .+ [p.stage_y, p.stage_x]
    centre_coords(x, h, w) = ((x .- ([h w] ./2)) .* [-1 1])'
    camera_M = [camera_m["a11"] camera_m["a12"]; camera_m["a21"] camera_m["a22"]]
    
    corr_coords = camera_M * centre_coords(hcat(y,x), fov.height, fov.width)
    translation(corr_coords, fov)'
end


function to_stage_coords(df::DataFrame,camera_m, img_metadata::Dict{String,Any})
    pixelmicrons = img_metadata["pixelmicrons"]
    
    fov = let p = img_metadata
        w,h = p["img_size"] |> size -> (size.x,size.y)
        stage_x = p["stage_pos_x"][end]
        stage_y = p["stage_pos_y"][end]
        FOV(h,w, pixelmicrons,stage_y,stage_x)
    end
    
    x = df.centroid_x
    y = df.centroid_y
    
    to_stage_coords(camera_m, x,y, fov)
end


#schema = Dict(
#    "query" => "Query",
#    "Query" => Dict(
#        "correlative" => "Correlative",
#        "fov" => "FOV"
#        ),
#    "FOV" => Dict(),
#    "Correlative" => Dict(
#        "centroid_x" => "Column",
#        "centroid_y" => "Column",
#        "last_x" => "Column",
#        "last_y" => "Column"
#        ),
#    "Column" => Dict(
#        "name" => "String"
#        )
#    )
#
#resolvers(state) = Dict(
#    "Query" => Dict(
#        "correlative" => (parent, args) -> collect_state(state,args),
#        "fov" => (parent,args) -> get_fovs(state, args)
#        )
#    )

function DDMFramework.query_state(state::CorrelativeState, query)
    println(query)
    #execute_query(query["query"], schema,resolvers(state)) |> JSON.json
    
    if sum(state.responses) > 2
        JSON.json(Dict())
    else
        println("Not enough FOVs have been matched.")
        JSON.json(Dict())
    end
end

#function Base.show(io::IO, mime::MIME"test/html", state::CorrelativeState)
#    show(io,mime,collect_objects(state))
#end

function readnd2(io)
    mktempdir() do d
        path = joinpath(d, "file.nd2")
        open(path, "w") do iow
            write(iow, read(io))
        end
        BioformatsLoader.bf_import(path)[1]
    end
end

function __init__()
    DDMFramework.register_mime_type("image/nd2", readnd2)
    DDMFramework.add_plugin("correlative", CorrelativeState)
end

export handle_update


end # module
