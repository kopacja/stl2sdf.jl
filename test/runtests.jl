using Test
using signedDistanceField
using NearestNeighbors
using LinearAlgebra
using DelimitedFiles

@testset "signedDistanceField.jl" begin

    #signedDistanceField.getVoxelSphere(1.0, 0.1, "sphere.vtk")

    # triangles = signedDistanceField.readSTL("CANTILEVER_BEAM_120_40_30_SC1_POST1000_LEVELSET_1_NOD_PID2.stl")
    # writedlm( "triangles.csv",  triangles, ',')
    #
    # println("number of triangles = ", length(triangles))
    # X = Vector{Vector{Float64}}()
    # push!(X, triangles[1][1])
    #
    # IEN = Vector{Vector{Float64}}()
    #
    # for triangle in triangles
    #     IEN_el = zeros(3)
    #     for i = 1:3
    #
    #         a = findfirst(x-> norm(x - triangle[i]) < 1.0e-5, X)
    #
    #         if (a == nothing)
    #             push!(X, triangle[i])
    #             IEN_el[i] = length(X)
    #         else
    #             IEN_el[i] = a[1]
    #         end
    #     end
    #     push!(IEN, IEN_el)
    # end
    # println("number of nodes = ", length(X))
    # writedlm( "X.csv",  X, ',')
    # writedlm( "IEN.csv",  IEN, ',')

    X = transpose(readdlm( "X_beam.csv", ','))
    nnp = size(X,2)

    IEN = transpose(readdlm( "IEN_beam.csv", ','))
    IEN = map(y->round(Int,y), IEN)

    INE = [Vector{Int64}() for _ = 1:length(X)]
    for el = 1:size(IEN,2)
        for i = 1:3
            push!(INE[IEN[i,el]], el)
        end
    end

    AABB_min = minimum(X, dims=2)
    AABB_max = maximum(X, dims=2)
    AABB_size = AABB_max .- AABB_min

    I = 50
    scale = 2.0
    voxelSize = minimum(AABB_size ./ I)

    M = Int.(ceil.( (AABB_max .- AABB_min) ./ voxelSize ) .+ 1)
    J =  floor.( (N.-1) .* (X.-AABBmin) ./ (AABBmax.-AABBmin));
    n = Int((J[1] + 1)*(J[2] + 1)*(J[3] + 1))

    points = zeros(3,n)
    a = 1
    for k = 0:J[3]
        for j = 0:J[2]
            for i = 0:J[1]
                points[:,a] = AABB_min .+ voxelSize * [i, j, k] .- voxelSize
                a += 1
            end
        end
    end
    writedlm( "points.csv",  transpose(points), ',')

    N = Int.(ceil.( (AABB_max .- AABB_min) ./ (scale*voxelSize) ) .+ 1)

    head = -1 * ones(Int64, prod(N));
    next = -1 * ones(Int64, nnp);

    I =  floor.( (N.-1) .* (points.-AABBmin) ./ (AABBmax.-AABBmin));
    I = I[3,:] .* (N[1]*N[2]) .+ I[2,:] .* N[1] .+ I[1,:] .+ 1

    for i ∈ [1:nnp]
        next[i] = head[I[i]];
        head[I[i]] = i;
    end

    # println("Init KDTree...")
    # kdtree = KDTree(X; leafsize = 10)
    # println("..done")
    # idxs, dists = knn(kdtree, points, 1, true)

    dists, xp = signedDistanceField.evalSignedDiscances(head, next, points, X, IEN, INE, idxs)
    # println("dists = ", dists)
    # println("xp = ", xp)

    writedlm( "xp.csv",  transpose(xp), ',')

    io = open("cube.vtk", "w")
    write(io, "# vtk DataFile Version 1.0\n")
    write(
        io,
        "Texture map for thresholding data (use boolean textures for 2D map)\n",
    )

    write(io, "ASCII\n\n")
    write(io, "DATASET STRUCTURED_POINTS\n")
    dim_x = Int(I[1] + 1)
    dim_y = Int(I[2] + 1)
    dim_z = Int(I[3] + 1)
    write(io, "DIMENSIONS $dim_x $dim_y $dim_z\n")
    write(io, "SPACING $voxelSize $voxelSize $voxelSize\n")

    org_x = AABB_min[1] - voxelSize
    org_y = AABB_min[2] - voxelSize
    org_z = AABB_min[3] - voxelSize
    write(io, "ORIGIN $org_x $org_y $org_z\n\n")

    write(io, "POINT_DATA $n\n")
    write(io, "SCALARS distance float 1\n")
    write(io, "LOOKUP_TABLE default\n")

    for dist ∈ dists
        if (abs(dist) > voxelSize)
            dist = sign(dist) * voxelSize
        end
        # dist = sign(dist)
        write(io, "$dist\n")
    end
    close(io)

end
