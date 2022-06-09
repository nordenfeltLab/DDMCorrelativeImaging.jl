
function compare_coordinates(comparison_data, coords; N_RANSAC::Int64 = 5000, n_triangles::Int64=1)
    transform_matrix = zeros(3,3)
    hit_indices = (Array{Float64}(undef,0,2), Array{Float64}(undef,0,2))
    best_hit_rate = 0.0
    best_score = 0.0
    fov_id = 0
    
    map(enumerate(comparison_data)) do (i, comparison)
        hits = find_triangle_matches(coords, comparison, n_triangles)
        hits_score = median(map(first, hits))
        
        if hits_score .< 0.01
            indices = hcat([[i,h[2]] for (i,h) in enumerate(hits)]...)
            coord_hits = comparison[indices[2,:],:]
            hit_rate,tmp_matrix = test_coordinate_transform(coord_hits,
                                                        coords,
                                                        N_RANSAC,
                                                        find_transform,
                                                        test_transform)
            if hit_rate .> best_hit_rate
                transform_matrix = tmp_matrix
                hit_indices = indices
                best_hit_rate = hit_rate
                best_score = hits_score
                fov_id = i
            end
        end
    end
    
    (M=transform_matrix, indices=hit_indices, hit_rate=best_hit_rate, score=best_score, fov_id=fov_id)
end

function find_transform(coord1,coord2)
    error_handler(e) = e isa SingularException ? [1 0 0; 0 1 0; 0 0 1] : rethrow()
    try
        return hcat(coord1, ones(size(coord1,1))) \ hcat(coord2, ones(size(coord2,1)))
    catch e
        return error_handler(e)
    end
end

function test_transform(coords, M)
    new_points = mapreduce(vcat, eachrow(coords)) do r
        r = [r...,1]'
        r * M
    end
    new_points[:,1:2]
end
        
function test_coordinate_transform(coord1, coord2, N, find_func, test_func; N_points = 3, lambda = 3)
    
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


function get_matches(centroids, comparison)
    hits = compare_coordinates(comparison, centroids)
    matched = if hits.hit_rate .> 0
        comparison[hits.fov_id][hits.indices[2,:], :]
    else
        []
    end
    return hits.hit_rate, centroids, matched
end


