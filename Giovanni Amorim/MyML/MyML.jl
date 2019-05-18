module MyML

function distance(v1,v2)
    d = 0
    for i=1:length(v1)
        d += (v1[i]-v2[i])^2
    end
    return d^0.5
end

function k_cluster(k,data)
    if length(data) <= k
        return data
    end
    C = Dict()
    for j in data[1:k]
        C[j] = [j]
    end
    prctg_dif = 10
    error_now = 0
    while prctg_dif > 0.01
        C = update_clusters(data,C)
        C = update_centers(C)
        e = in_error(C)
        prctg_dif = abs(e - error_now)/error_now
        error_now = e
    end
    # result_C = Dict()
    # for center in keys(C)
    #     total_dists = sum([distance(center,t) for t in C[center]])
    #     result_C[center] = []
    #     for t in C[center]
    #         push!(result_C[center], (t,distance(center,t)/total_dists))
    #     end
    # end
    return [c for c in keys(C)]
end

function in_error(C)
    Ein = 0
    for center in keys(C)
        for point in C[center]
            Ein = distance(center,point)^2
        end
    end
    return Ein
end

function update_clusters(data, C)
    points_closest_centers = Dict()
    for point in data
        min_dist = 10^10
        for center in keys(C)
            dist = distance(point,center)
            if dist < min_dist
                min_dist = dist
                points_closest_centers[point] = center
            end
        end
    end
    for center in keys(C)
        C[center] = []
        for point in data
            if points_closest_centers[point] == center
                push!(C[center], point)
            end
        end
    end
    return C
end

function update_centers(C)
    new_C = Dict()
    for cluster in values(C)
        new_center = sum(cluster)/length(cluster)
        new_C[new_center] = cluster
    end
    return new_C
end

function myrand_bool(N,dimension)
    results = []
    for n=1:N
        r = zeros(dimension)
        for d=1:dimension
            seed = rand()
            if seed >= 0.5
                r[d] = 1
            else
                r[d] = -1
            end
        end
        push!(results,r)
    end
    return results
end

function ajust_rand_bounds(n,lower,upper)
    return n*(upper-lower) + lower
end

function myrand_float(N,dimension,low,up)
    results = []
    for n=1:N
        push!(results, ajust_rand_bounds.(rand(dimension),low,up))
    end
    return results
end

end
