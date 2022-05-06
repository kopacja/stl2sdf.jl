module signedDistanceField

using LinearAlgebra

function getVoxelSphere(
    R::Float64,
    voxelSize::Float64,
    fileName::String,
)::Vector{Float64}
    len = 2 * (R + voxelSize)
    center = 0.5 * len * ones(3)
    I = ceil(len / voxelSize)
    n = Int((I + 1)^3)
    D = zeros(n)

    io = open(fileName, "w")
    write(io, "# vtk DataFile Version 1.0\n")
    write(
        io,
        "Texture map for thresholding data (use boolean textures for 2D map)\n",
    )
    write(io, "ASCII\n\n")

    write(io, "DATASET STRUCTURED_POINTS\n")
    dim = Int(I + 1)
    write(io, "DIMENSIONS $dim $dim $dim\n")
    write(io, "SPACING $voxelSize $voxelSize $voxelSize\n")
    write(io, "ORIGIN 0.0 0.0 0.0\n\n")

    write(io, "POINT_DATA $n\n")
    write(io, "SCALARS distance float 1\n")
    write(io, "LOOKUP_TABLE default\n")

    a = 1
    for i = 0:I
        for j = 0:I
            for k = 0:I
                X = voxelSize * [i, j, k]
                dist = norm(X - center) - R
                if (abs(dist) > sqrt(2 * voxelSize))
                    dist = sign(dist) * sqrt(2 * voxelSize)
                end
                D[a] = dist
                a = a + 1
                write(io, "$dist\n")
            end
        end
    end
    close(io)
    return D
end

function readSTL(fileName::String)
    normals = Vector{Vector{Float64}}()
    triangles = Vector{Vector{Vector{Float64}}}()

    fileID = open(fileName)
    s = readline(fileID) # "solid"
    s = split(readline(fileID)) # "facet" of "endsolid"
    while (s[1] != "endsolid")
        normal = zeros(3)
        normal[1] = parse(Float64, s[3])
        normal[2] = parse(Float64, s[4])
        normal[3] = parse(Float64, s[5])
        s = readline(fileID) # outer loop
        triangle = [Vector{Float64}(undef, 3) for _ = 1:3]
        for i ∈ 1:3
            s = split(readline(fileID))
            triangle[i][1] = parse(Float64, s[2])
            triangle[i][2] = parse(Float64, s[3])
            triangle[i][3] = parse(Float64, s[4])
        end
        push!(normals, normal)
        push!(triangles, triangle)

        s = readline(fileID) # "endloop"
        s = readline(fileID) # "endfacet"
        s = split(readline(fileID)) # "facet" of "endsolid"
    end
    close(fileID)
    return triangles
end

function evalSignedDiscances(head::AbstractArray, next::AbstractArray, points::AbstractArray, X::AbstractArray, IEN::Matrix{Int64}, INE::Vector{Vector{Int64}}, idxs::AbstractArray)

    nsd = size(points, 1)
    nnp = size(points, 2)

    dists = zeros(nnp)
    xp = zeros(nsd, nnp)
    n_avrg = zeros(nsd)

    for i = 1:nnp
        isInside = false;
        dist = 1.0e32
        n = zeros(nsd)
        x = points[:,i]
        for j = 1:length(INE[idxs[i][1]])
            el = INE[idxs[i][1]][j]
            x₁ = X[:,IEN[1, el]]
            x₂ = X[:,IEN[2, el]]
            x₃ = X[:,IEN[3, el]]
            n = cross(x₂-x₁, x₃-x₁)

            n = n / norm(n)
            n_avrg += n

            A = [(x₁[2]*n[3] - x₁[3]*n[2])  (x₂[2]*n[3] - x₂[3]*n[2])   (x₃[2]*n[3] - x₃[3]*n[2])
                 (x₁[3]*n[1] - x₁[1]*n[3])  (x₂[3]*n[1] - x₂[1]*n[3])   (x₃[3]*n[1] - x₃[1]*n[3])
                 (x₁[1]*n[2] - x₁[2]*n[1])  (x₂[1]*n[2] - x₂[2]*n[1])   (x₃[1]*n[2] - x₃[2]*n[1])]
            b = [x[2]*n[3] - x[3]*n[2], x[3]*n[1] - x[1]*n[3], x[1]*n[2] - x[2]*n[1]]

            n_max, i_max = findmax(abs.(n))
            A[i_max,:] = [1.0 1.0 1.0]
            b[i_max] = 1.0
            λ = A\b

            if (minimum(λ) >= 0.0)
                isInside = true
                xₚ = λ[1]*x₁ + λ[2]*x₂ + λ[3]*x₃
                xp[:,i] = xₚ
                dist_tmp =  dot(x - xₚ,n)
                if(abs(dist_tmp) < abs(dist))
                    dist = dist_tmp
                    # println("Lambda!")
                end
            end
        end

        if (!isInside)
            n_avrg = n_avrg / length(INE[idxs[i][1]])
# println("!isInside")
            xₚ = X[:,idxs[i][1]]
            #dist = sign(dot(x - xₚ,n)) * norm(x - xₚ)
            dist = 1.0e32
            # Loop over edges:
            for j = 1:length(INE[idxs[i][1]])
                el = INE[idxs[i][1]][j]
                x₁ = X[:,IEN[1, el]]
                x₂ = X[:,IEN[2, el]]
                x₃ = X[:,IEN[3, el]]
                e₁ = x₂-x₁
                e₂ = x₃-x₂
                e₃ = x₁-x₃

                n = cross(e₁, e₂)
                n = n / norm(n)
                n_avrg += n

                L₁ = norm(e₁)
                L₂ = norm(e₂)
                L₃ = norm(e₃)

                P₁ = dot(x-x₁, e₁/L₁)
                P₂ = dot(x-x₂, e₂/L₂)
                P₃ = dot(x-x₃, e₃/L₃)

                if(P₁ >= 0 && P₁ <= L₁)
                    xₚ = x₁ + e₁/L₁*P₁
                    dist_tmp = sign(dot(x - xₚ,n)) * norm(x - xₚ)
                    if(abs(dist_tmp) < abs(dist))
                        dist = dist_tmp
                        # println("el = ", el)
                        # println("P₁ = ", P₁)
                        # println("n = ", n)
                    end
                end

                if(P₂ >= 0 && P₂ <= L₂)
                    xₚ = x₂ + e₂/L₂*P₂
                    dist_tmp = sign(dot(x - xₚ,n_avrg)) * norm(x - xₚ)
                    if(abs(dist_tmp) < abs(dist))
                        dist = dist_tmp
                    end
                end

                if(P₃ >= 0 && P₃ <= L₃)
                    xₚ = x₃ + e₃/L₃*P₃
                    dist_tmp = sign(dot(x - xₚ,n_avrg)) * norm(x - xₚ)
                    if(abs(dist_tmp) < abs(dist))
                        dist = dist_tmp
                    end
                end
            end
            xp[:,i] = xₚ
        end
        dists[i] = dist
    end
    return dists, xp
end

end # module
