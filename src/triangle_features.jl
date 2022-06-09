function triangle_vectorize(points)
    p1 = points[1,:]
    p2 = points[2,:]
    p3 = points[3,:]
    
    v1 = p2-p1
    v2 = p3-p1
    
    v2l = norm(v2)
    vb = dot(v1,v2) / v2l^2
    vh = sqrt(norm(v1)^2 - vb^2)/v2l
    return hcat(vb,vh)
end

findextrema(v,n;rev=false) = partialsortperm(v,1:n,rev=rev)
function get_triangle_features(points; n_triangles = 0)
    dist = Distances.pairwise(Euclidean(), points', points')
    idx = [findextrema(r, 3+n_triangles) for r in eachrow(dist)]
    
    features = Array{Float64,2}[]
    
    for t in idx
        for i in 1:n_triangles
            idx = t[[1,1+i,2+i]]
            coords = map(x -> points[x,1:2],idx)
            push!(features, triangle_vectorize(coords))
        end
    end
    
    return repeat(points, inner = (n_triangles,1)), vcat(features...)
end

function find_triangle_matches(p1,p2, n_triangles = 1)
    p1, p1_features = get_triangle_features(p1, n_triangles = n_triangles) 
    p2, p2_features = get_triangle_features(p2, n_triangles = n_triangles) 
    feature_distances = Distances.pairwise(Euclidean(), p1_features', p2_features')
    hist = [findmin(r) for r in eachrow(feature_distances)]
    return hist
end
    