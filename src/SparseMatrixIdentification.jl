module SparseMatrixIdentification
using LinearAlgebra
using SparseArrays
using BandedMatrices

# compute the percentage banded for a matrix given a bandwidth
function compute_bandedness(A, bandwidth)

    if bandwidth == 0 
        return 100
    end
        
    n = size(A, 1)
    total_band_positions = 0
    non_zero_in_band = 0
    bandwidth = bandwidth
    for r in 1:n
       
        for c in 1:n
            if abs(r - c) < bandwidth
                print("R", r)
                print("C", c)
                total_band_positions += 1  # This position belongs to the band
                if A[r, c] != 0
                    non_zero_in_band += 1  # This element is non-zero in the band
                end
            end
        end
    end

    percentage_filled = non_zero_in_band / total_band_positions * 100
    return percentage_filled
end

function is_banded(A, bandwidth)
    n = size(A, 1)  # assuming A is square

    # Count the number of non-zero entries outside the band
    non_band_nonzeros = 0
    for r in 1:n
        for c in 1:n
            if abs(r - c) >= bandwidth && A[r, c] != 0
                non_band_nonzeros += 1
            end
        end
    end
    
    # If there are any non-zero entries outside the band, it's not banded
    return non_band_nonzeros == 0
end

# compute the sparsity for a given matrix
function compute_sparsity(A)
    n = size(A, 1)
    percentage_sparsity = length(nonzeros(A)) / n^2
    return 1 - percentage_sparsity
end

export getstructure

# get the percentage banded for a bandwidth of 1 and percentage sparsity
function getstructure(A::Matrix)::Any
    percentage_banded = compute_bandedness(A, 1)
    percentage_sparsity = compute_sparsity(SparseMatrixCSC(A))

    return (percentage_banded, percentage_sparsity)
end

export sparsestructure

# return the best type of matrix for a given sparse matrix
function sparsestructure(A::SparseMatrixCSC)::Any
    sym = issymmetric(A)
    herm = ishermitian(A)
    banded = is_banded(A,2)
    posdef = isposdef(A)
    lower_triangular = istril(A)
    upper_triangular = istriu(A)


    n = size(A, 1)
    
    if sym 
        return Symmetric(A)
    end 

    if herm
        return Hermitian(A)
    end

    if lower_triangular
        return LowerTriangular(A)
    end

    if upper_triangular
        return UpperTriangular(A)
    end

    if banded
        return BandedMatrix(A)
    end

    return SparseMatrixCSC(A)
end

end


