# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Julia 1.5.1
#     language: julia
#     name: julia-1.5
# ---

# +

struct Matched
    hit_rate::Float64
    points::Matrix{Float64}
    matched_points::Matrix{Float64}
end

struct ComparisonData
    arr::Vector{Matrix{Float64}}
    df::DataFrame
end
#mutable struct Results
#    df::DataFrame
#    m::Matrix{Float64}
#end
struct ComparativeState
    matched::Vector{Matched}
    comparison_data::ComparisonData
    responses::Vector{Int64}
#    results::Results
end

assay_id = 946
well = 3
function init()

    #data stored through "../../dev/migration/image_analysis.ipynb" and "../../dev/migration/data_analysis.ipynb"
    
    comparison_arr = load("../../DDM/dev/migration/$(assay_id)_well_$(well)_analyzed.jld2", "comparison_arr")
    comparison_dataset = load("../../DDM/dev/migration/$(assay_id)_well_$(well)_analyzed.jld2", "speed_groups")
    
    #select data - in this case from migration
#    groups = [1, 10]
#    results = let speed_groups = filter(x -> x.speed_group ∈ groups, comparison_dataset)
#        combine(groupby(speed_groups, :speed_group)) do df
#            sample_df(df, 50)#req["n"] * size(groups,1)
#        end
#    end
    #for TIRF to SIM
    results = load("../../DDM/dev/migration/experiment_136.jld2", "results")
    
    #this should be in the get_results function but because im saving the state in update(), and i need to have an easy way to look at what i want to image (output is slightly different from other modules), this is the easiest way
    ComparativeState([],ComparisonData(comparison_arr, results), [])
end

Base.size(a::ComparativeState, args...) = size(a.matched, args...)
Base.push!(a::ComparativeState, b::Matched) = push!(a.matched, b)



function update(img, params, state::ComparativeState)
#    img = ShadingCorrection.correct_for_shading(img, "DAPI", "10X", microscope = "TIRF")
    matches = compare_image(img, params, state.comparison_data.arr)
    
    display("hitrate: $(matches[1])")
    status = if matches[1] > 1/3
        push!(state, Matched(matches...))
        push!(state.responses, 1)
        1
    else
        push!(state.responses, 0)
        0
    end
    
    return status, state
end


function get_results(state::ComparativeState, req)
    display(state.responses)
    
    status, results = if size(state.matched,1) >= 3
        display("estimating matrix")
        # best way to get the most accurate transform matrix as possible??
        hit_rate, M = get_correlation_matrix(state)
        results = let results_df = state.comparison_data.df
            let c = hcat(results_df.relative_y, results_df.relative_x)
                transformed = hcat(c, ones(size(c,1))) * M
                DataFrame(
                    "stage_pos_y" => transformed[:,1],
                    "stage_pos_x" => transformed[:,2],
                    "centroid_y" => 0, #fill(1192/2, size(transformed,1)),
                    "centroid_x" => 0 #fill(1192/2, size(transformed,1))
                )
            end
        end
        
        true, results
    elseif last(state.responses) == 1
        display("we found a point")
        results = DataFrame(
                "stage_pos_y" => 0,
                "stage_pos_x" => 0,
                "centroid_y" => 0,
                "centroid_x" => 0
            )
        true, results
    else
        false, DataFrame()
    end
    
    return status, results
end

function rm_touching_border(img_lb)
    to_remove = unique(hcat(img_lb[1,:], img_lb[end,:], img_lb[:,1], img_lb[:, end]))
    to_remove = to_remove[to_remove .!== 0]
    [x ∈ to_remove ? 0 : x for x in img_lb]
end

function nuclei_centroids(img; sigma = 5, minsize = 100, maxsize = Inf)

    img_lb = Common.snr_binarize(img, sigma = sigma) |> label_components |> rm_touching_border
    
    objects = filter(x-> x.second .> minsize, countmap(img_lb[:]))
    centroids = [RegionProps.centroid(img, ind) for in in keys(objects)]
    #centroids = sparse(img_lb) |> img -> mapreduce(x->centroid(img, x), vcat, collect(keys(objects)))
end
nuclei_centroids(img, seg_params) = nuclei_centroids(img, sigma = seg_params["snr"], minsize = seg_params["minsize"], maxsize = seg_params["maxsize"])


label_indices(labeled, index::Int64) = findall(labeled .== index)

function centroid(im, index::Int)
    ind = label_indices(im, index)
    ind_array = [i[j] for i in ind, j in 1:2]
    return hcat(mean(ind_array[:,1]),mean(ind_array[:,2]))
end



findextrema(v, n; rev=false) = partialsortperm(v, 1:n; rev=rev)


function triangle_vectorize(points)
    p1 = points[1,:]
    p2 = points[2,:]
    p3 = points[3,:]

    v1 = p2-p1
    v2 = p3-p1

    v2l = norm(v2)
    vb = dot(v1,v2) / v2l^2
#    v_tmp = abs()
    vh = sqrt(norm(v1)^2 - vb^2)/v2l
    return hcat(vb, vh)
end

function get_triangle_features(points; n_triangles = 0)
    dists = Distances.pairwise(Euclidean(), points', points')
    tri_idx = [findextrema(r,3+n_triangles) for r in eachrow(dists)]
    
    tri_features = Array{Float64,2}[]
    
    for t in tri_idx
        for i in 1:n_triangles
            idx = t[[1,1+i,2+i]]
            tri_coords = map(x -> points[x,1:2], idx)
            out = triangle_vectorize(tri_coords)
            tri_features = push!(tri_features, out)
        end
    end
    
    return repeat(points, inner = (n_triangles,1)), vcat(tri_features...)
end

function find_triangle_matches(p1, p2;n_triangles = 1)
    p1, p1_features = get_triangle_features(p1, n_triangles = n_triangles)
    p2, p2_features = get_triangle_features(p2, n_triangles = n_triangles)

    feat_dists = Distances.pairwise(Euclidean(), p1_features', p2_features')
    hits = [findmin(r) for r in eachrow(feat_dists)]
    return hits
end




FOV_params(h,w,p,y,x,i) = (height = h, width = w, pixelmicrons = p, stage_y = y, stage_x = x, index = i)
FOV_params(m::Dict{String, Any}; index = 1, pixelmicrons = m["pixelmicrons"]) = FOV_params(m["height"], m["width"], pixelmicrons, m["stage_pos_y"][index], m["stage_pos_x"][index], index)

parse_camera_angle(cp) = [cp["a11"] cp["a12"] ; cp["a21"] cp["a22"]]



#centre_coords(x, h, w) = ((x .- ([h w] ./2)) .* [-1 1])'
#function to_stage_coords(camera_M, c::Array{Float64, 2}, pixelmicrons, stage_y, stage_x, ;height=2424, width=2424)
#    translation(x, p) =  x .* p.pixelmicrons .+ [p.y p.x]
#    
#    p = (pixelmicrons=pixelmicrons, y=stage_y, x=stage_x, h=height, w=width)
#    
#    corr_coords = camera_M * centre_coords(c, p.h, p.w)
#    translation(corr_coords', p)
#end

#to_stage_coords(camera_M, c::Array{Float64, 2}, p) = to_stage_coords(camera_M, c::Array{Float64, 2}, p.pixelmicrons, p.stage_y, p.stage_x, height=p.height, width=p.width)

function compare_image(img, params, comparison_data)
    
    microscope = params["system_parameters"]["microscope"]
    camera_params = JSON.Parser.parsefile("../../DDM/CorrelativeImaging/src/camera_angle_$(microscope).json")
    camera_M  = parse_camera_angle(camera_params)
    
    h,w = size(img)
    
    params["experiment_parameters"]["pixelmicrons"] = camera_params["pixelmicrons"]
    params["experiment_parameters"]["height"] = h
    params["experiment_parameters"]["width"] = w
    fov_params = FOV_params(params["experiment_parameters"])
    
    params["segmentation"] = Dict("snr" => 5, "minsize" => 100, "maxsize" => Inf)
    
    centroids_img = nuclei_centroids(img, params["segmentation"])
    
    matches = let points = to_stage_coords(camera_M, centroids_img, fov_params)
        get_matches(points, comparison_data)
    end
end


function get_matches(centroids, comparative_dataset)
    hits = compare_centroids(comparative_dataset, centroids)
    matched = if hits.hit_rate .> 0
        comparative_dataset[hits.file_i][hits.indices[2,:], :]
    else
        []
    end
    return hits.hit_rate, centroids, matched
end


function get_correlation_matrix(state::ComparativeState)
    
    N = 10000
    max_distance = 10
    
    points = mapreduce(x -> x.points, vcat, state.matched)
    matched_points = mapreduce(x -> x.matched_points, vcat, state.matched)
    
    hit_rate, M = let (c1, c2) = (matched_points, points)
        transform_coordinates_test(c1, c2, N, find_transform, test_transform, N_points = 3, lambda = max_distance)
    end
    return hit_rate, M
end


function compare_centroids(comparison_data, centroids; N_RANSAC = 5000)
    transform_matrix = zeros(3,3)
    hit_indices = (Array{Float64}(undef, 0, 2),Array{Float64}(undef, 0, 2))
    best_hit_rate = 0.0
    best_score = 0.0
    file_i = 0
    
    map(enumerate(comparison_data)) do (i, comparison)

        hits = find_triangle_matches(centroids, comparison, n_triangles = 1)
        hits_score = median(map(x -> x[1], hits))

        if hits_score .< 0.01
            indices = hcat([[i, h[2]] for (i,h) in enumerate(hits)]...)
            coord_hits = comparison[indices[2,:],:]
            hit_rate, temp_matrix = transform_coordinates_test(coord_hits, centroids, N_RANSAC, find_transform, test_transform)
            
            if best_hit_rate .< hit_rate
                transform_matrix = temp_matrix
                hit_indices = indices
                best_hit_rate = hit_rate
                best_score = hits_score
                file_i = i
            end
#                println("transform success rate: $(round(Int64, 100*hit_rate)) %")
        end
    end
    
    (M = transform_matrix, indices = hit_indices, hit_rate = best_hit_rate, score = best_score, file_i = file_i)
end

function find_transform(coord1, coord2)
    error_handler(e) = e isa SingularException ? [1 0 0 ; 0 1 0 ; 0 0 1] : rethrow()

    try 
        return hcat(coord1, ones(size(coord1,1))) \ hcat(coord2, ones(size(coord2,1)))
    catch e
        return error_handler(e)
    end
end 

function test_transform(coord1, M)
    new_points = map(eachrow(coord1)) do r
        r = [r..., 1]'
        r * M
    end
    new_points = vcat(new_points...)
    return new_points[:,1:2]
end

function transform_coordinates_test(coord1, coord2, N, find_func, test_func; N_points = 3, lambda = 3)
    
    M_init = zeros(Float64, 3,3)
    score = (0, M_init)
    for n in 1:N
        rand_i = rand(1:size(coord1,1), N_points)
        M = find_func(coord1[rand_i, :], coord2[rand_i, :])
        new_points = test_func(coord1, M)
        dists = Distances.colwise(Euclidean(), new_points', coord2')
        hits = count(dists .< lambda)
        score = hits > score[1] ? (hits,M) : score
    end

    return score[1] / size(coord2,1), score[2]
end