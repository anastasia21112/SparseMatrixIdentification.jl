# using SparseMatrixIdentification
# using Test

# @testset "SparseMatrixIdentification.jl" begin
#     # Write your tests here.
# end
using SparseMatrixIdentification
using Test
using LinearAlgebra
using SparseArrays
using BandedMatrices
using ToeplitzMatrices
using BlockBandedMatrices

@testset "Test check_diagonal" begin
    A = [1 2 3; 4 1 5; 6 7 1]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # Diagonal starting at (1, 1) should be all 1s
    @test SparseMatrixIdentification.check_diagonal(A, 1, 2) == false  # Diagonal starting at (1, 2) should not be all 1s

    # Test 2: Edge case with a 1x1 matrix
    A = [7]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # A 1x1 matrix is trivially a Toeplitz matrix

    # Test 3: Negative case, diagonal mismatch
    A = [1 2 3; 4 1 5; 6 7 2]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == false  # Diagonal starting at (1, 1) should be different

    # Test 4: Larger matrix, with valid Toeplitz diagonals
    A = [1 2 3 4; 5 1 2 3; 6 5 1 2; 7 6 5 1]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # Diagonal starting at (1, 1) should be all 1s
    @test SparseMatrixIdentification.check_diagonal(A, 1, 2) == true   # Diagonal starting at (1, 2) should be all 2s

    # Test 5: Non-square matrix, checking diagonals in both directions
    A = [1 2 3; 4 1 2; 5 4 1]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == true   # Diagonal starting at (1, 1) should be all 1s
    @test SparseMatrixIdentification.check_diagonal(A, 1, 2) == true   # Diagonal starting at (1, 2) should be all 2s

    # Test 6: Full mismatch
    A = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.check_diagonal(A, 1, 1) == false  # Diagonal starting at (1, 1) should not match, as 1 != 4
end

@testset "Test is_toeplitz" begin
    # Test 1: A basic Toeplitz matrix (2x2)
    mat1 = [1 2; 3 1]
    @test SparseMatrixIdentification.is_toeplitz(mat1) == true
    
    # Test 2: A non-Toeplitz matrix (3x3)
    mat2 = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.is_toeplitz(mat2) == false
    
    # Test 3: A 1x1 matrix (Trivially Toeplitz)
    mat3 = [5]
    @test SparseMatrixIdentification.is_toeplitz(mat3) == true
    
    # Test 4: A 3x3 Toeplitz matrix
    mat4 = [1 2 3; 4 1 2; 5 4 1]
    @test SparseMatrixIdentification.is_toeplitz(mat4) == true
    
    # Test 5: A 3x3 non-Toeplitz matrix with different diagonals
    mat5 = [1 2 3; 4 1 5; 6 7 1]
    @test SparseMatrixIdentification.is_toeplitz(mat5) == false
    
    # Test 6: A matrix with identical columns, which is not Toeplitz
    mat6 = [1 1 1; 2 2 2; 3 3 3]
    @test SparseMatrixIdentification.is_toeplitz(mat6) == false
    
    # Test 7: A 2x2 Toeplitz matrix
    mat7 = [1 2; 2 1]
    @test SparseMatrixIdentification.is_toeplitz(mat7) == true
    
    # Test 9: A large Toeplitz matrix (5x5)
    mat8 = [
        1 2 3 4 5;
        6 1 2 3 4;
        7 6 1 2 3;
        8 7 6 1 2;
        9 8 7 6 1
    ]
    @test SparseMatrixIdentification.is_toeplitz(mat8) == true
    
    # Test 10: A large non-Toeplitz matrix (5x5)
    mat9 = [
        1 2 3 4 5;
        6 7 8 9 10;
        11 12 13 14 15;
        16 17 18 19 20;
        21 22 23 24 25
    ]
    @test SparseMatrixIdentification.is_toeplitz(mat9) == false
end


# Test for `compute_bandedness` function
@testset "Test compute_bandedness" begin
    # Test 1: Identity Matrix
    A = Matrix(I, 3, 3)
    bandwidth = 1
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0

    # Test 2: Band matrix (with zeros outside the band)
    A = [1 2 0; 3 4 5; 0 6 7]
    bandwidth = 1
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0

    # Test 3: Sparse random matrix
    A = [1 0 0; 0 2 0; 0 0 3]
    bandwidth = 0
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0

    # Test 4: Full random matrix (bandwidth = 1, not banded)
    A = [1 2 3; 4 5 6; 7 8 9]
    bandwidth = 1
    @test SparseMatrixIdentification.compute_bandedness(A, bandwidth) == 100.0 
end

# Test for `is_banded` function
@testset "Test is_banded" begin
    # Test 1: matrix with filled elements within the band
    A = [1 2 0; 3 4 5; 0 6 7]
    @test SparseMatrixIdentification.is_banded(A, 1/3) == false

    # Test 2: Identity matrix (banded)
    A = Matrix(I, 3, 3)
    @test SparseMatrixIdentification.is_banded(A, 1/3) == true

    # Test 3: Full random matrix (non-banded)
    A = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.is_banded(A, 1/3) == false
end

# Test for `compute_sparsity` function
@testset "Test compute_sparsity" begin
    # Test 1: Identity matrix
    A = Matrix(I, 3, 3)
    A = SparseMatrixCSC(A)
    @test SparseMatrixIdentification.compute_sparsity(A) == 1 - 1/3

    # Test 2: Sparse matrix with a few non-zero elements
    A = [0 0 0; 0 5 0; 0 0 0]
    A = SparseMatrixCSC(A)
    @test SparseMatrixIdentification.compute_sparsity(A) == 1- 1/9  # sparsity = 1 non-zero element / 9 total elements

    # Test 3: Full random matrix (sparsity = 0)
    A = [1 2 3; 4 5 6; 7 8 9]
    A = SparseMatrixCSC(A)
    @test SparseMatrixIdentification.compute_sparsity(A) == 0.0  # no sparsity
end

@testset "Test is_block_banded" begin
    # Test 1: Block banded matrix (2x2 blocks, bandwidth = 1)
    A1 = [
        1 2 0 0;
        3 4 5 0;
        0 6 7 8;
        0 0 9 10
    ]
    @test SparseMatrixIdentification.is_block_banded(A1, 2, 1) == false

    # Test 2: Block banded matrix (2x2 blocks, bandwidth = 2)
    A2 = [
        1 2 0 0 0;
        3 4 5 0 0;
        0 6 7 8 0;
        0 0 9 10 11;
        0 0 0 12 13
    ]
    @test SparseMatrixIdentification.is_block_banded(A2, 2, 2) == true

    # Test 3: Non-block banded matrix (no block structure)
    A3 = [
        1 2 3 4;
        5 6 7 8;
        9 10 11 12;
        13 14 15 16
    ]
    @test SparseMatrixIdentification.is_block_banded(A3, 2, 1) == false

    # Test 4: Block banded matrix (1x1 blocks, bandwidth = 0)
    A4 = [
        1 0 0 0;
        0 1 0 0;
        0 0 1 0;
        0 0 0 1
    ]
    @test SparseMatrixIdentification.is_block_banded(A4, 1, 0) == false

    # Test 5: Matrix with blocks that are outside the bandwidth
    A5 = [
        1 2 0 0;
        3 4 5 0;
        0 0 7 8;
        0 0 0 10
    ]
    @test SparseMatrixIdentification.is_block_banded(A5, 2, 1) == false

    # Test 6: Matrix with non-zero elements outside the expected block band
    A6 = [
        1 2 0 0 0;
        3 4 5 0 0;
        0 6 7 8 0;
        0 0 0 0 0;
        0 0 9 10 11
    ]
    @test SparseMatrixIdentification.is_block_banded(A6, 2, 2) == true

    # Test 7: Small 1x1 matrix
    A7 = [1]
    @test SparseMatrixIdentification.is_block_banded(A7, 1, 0) == true

    # Test 8: Empty matrix (edge case)
    A8 = Int[]
    @test SparseMatrixIdentification.is_block_banded(A8, 1, 0) == true
end

# Test for `getstructure` function
@testset "Test getstructure" begin
    # Test 1: Band matrix
    A = [1 2 0; 3 4 5; 0 6 7]
    @test getstructure(A) == (100.0, 0.2222222222222222)

    # Test 2: Identity matrix
    A = Matrix(I, 3, 3)
    @test getstructure(A) == (100.0, 0.6666666666666667)
end

# Test for `sparsestructure` function
@testset "Test sparsestructure" begin
 
    # Test 1: Sparse banded matrix (banded)
    A = [ 1  2  0; 3  4  5; 0  6  7 ]
    @test sparsestructure(sparse(A), band_threshold = 2/3) isa BandedMatrix

    # Test 2: Symmetric matrix
    A = [1 2 2; 2 3 4; 2 4 5]
    @test sparsestructure(SparseMatrixCSC(A)) isa Symmetric

    # Test 3: Hermitian matrix (complex conjugate symmetry)
    A = [1 2+3im 4+5im; 2-3im 6 7+8im; 4-5im 7-8im 9]
    @test sparsestructure(SparseMatrixCSC(A)) isa Hermitian

    # Test 4: Lower triangular matrix
    A = [1 0 0; 2 3 0; 4 5 6]
    @test sparsestructure(SparseMatrixCSC(A)) isa LowerTriangular

    # Test 5: Upper triangular matrix
    A = [1 2 3; 0 4 5; 0 0 6]
    @test sparsestructure(SparseMatrixCSC(A)) isa UpperTriangular

    # Test 6: Generic sparse matrix (fallback)
    B = [1 2 3; 4 5 6; 7 8 9]
    sparse_B = SparseMatrixCSC(B)
    @test sparsestructure(sparse_B) isa SparseMatrixCSC

    # Test 7: Toeplitz Sparse Matrix
    T = [1 2 0; 0 1 2; 0 0 1]
    sparse_T = SparseMatrixCSC(T)
    @test sparsestructure(sparse_T) isa Toeplitz

    # Test 8: Block Banded Matrix
    B = [1 1 0 0 0 0 0 0;
        1 1 1 1 0 0 0 0;
        1 1 1 1 1 1 0 0;
        0 0 1 1 1 1 1 0;
        0 0 0 1 1 1 1 1;
        0 0 0 0 1 1 1 1;
        0 0 0 0 0 1 1 1;
        0 0 0 0 0 0 1 1]
    sparse_B = SparseMatrixCSC(B)
    @test sparsestructure(sparse_B, block_threshold=2, band_threshold=1/3) isa BlockBandedMatrix
   
end

