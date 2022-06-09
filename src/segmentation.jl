
function sigma_segment(img;n_sigma=3.0,kwargs...)
    m,s = sigma_clipped_stats(img)
    img .> (m + (s * n_sigma))
end
otsu_close_seg(img) = closing(img .> otsu_threshold(img))

segmentation_lib = Dict(
    "sigma_segment" => (img;kwargs...) -> sigma_segment(img;kwargs...),
    "otsu_close" => (img,kwargs...) -> otsu_close_seg(img)
)

function segmentation_handler(img;method="sigma_segment",minsize=150,maxsize=2000, kwargs...)
    lb = segmentation_lib[method](img;kwargs...) |>
        label_components
    sp = sparse(lb)
    counts = countmap(nonzeros(sp))
    for (i,j,v) in zip(findnz(sp)...)
        if counts[v] < minsize || counts[v] > maxsize
            sp[i,j] = 0
        end
    end
    dropzeros!(sp)
    data = ((;r...) for r in regionprops(img,sp,selected=unique(nonzeros(sp))))
    DataFrame(data)
end
    