module SparseMatrixIdentification
using LinearAlgebra
using SparseArrays
using BandedMatrices
using ToeplitzMatrices


# check the diagonal of a given matrix, helper for is_toeplitz
function check_diagonal(A, i, j)
    N = size(A, 1)
    M = size(A, 2)

    num = A[i, j]
    i += 1
    j += 1


    while i <= N && j <= M
        if A[i, j] != num
            return false
        end
        i += 1
        j += 1
    end
    return true
end

# check if toeplitz matrix
function is_toeplitz(mat)
    N = size(mat, 1)  
    
    if N == 1
        return true
    end

    M = size(mat, 2)

    
    for j in 1:M
        if !check_diagonal(mat, 1, j)
            return false
        end
    end

    for i in 2:N
        if !check_diagonal(mat, i, 1)
            return false
        end
    end

    return true
end

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

function is_banded(A, threshold)
    n = size(A, 1)  # assuming A is square
    bandwidth = n * threshold
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
function sparsestructure(A::SparseMatrixCSC, threshold)::Any
    sym = issymmetric(A)
    herm = ishermitian(A)
    banded = is_banded(A, threshold)
    posdef = isposdef(A)
    lower_triangular = istril(A)
    upper_triangular = istriu(A)
    toeplitz = is_toeplitz(A)

    n = size(A, 1)
    
    if toeplitz
        first_row = A[1, :]
        first_col = A[:, 1]
        return Toeplitz(first_col, first_row)
    end
    
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


