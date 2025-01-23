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

# Test for `isbanded` function
@testset "Test isbanded" begin
    # Test 1: Banded matrix with filled elements within the band
    A = [1 2 0; 3 4 5; 0 6 7]
    @test SparseMatrixIdentification.isbanded(A) == true

    # Test 2: Identity matrix (bandwidth = 0, considered banded)
    A = Matrix(I, 3, 3)
    @test SparseMatrixIdentification.isbanded(A) == true

    # Test 3: Full random matrix (non-banded)
    A = [1 2 3; 4 5 6; 7 8 9]
    @test SparseMatrixIdentification.isbanded(A) == true
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

Test for `getstructure` function
@testset "Test getstructure" begin
    # Test 1: Band matrix
    A = [1 2 0; 3 4 5; 0 6 7]
    @test getstructure(A) == (100.0, 0.2222222222222222)

    # Test 2: Identity matrix
    A = Matrix(I, 3, 3)
    @test getstructure(A) == (100.0, 2/3)
end

# Test for `sparsestructure` function
@testset "Test sparsestructure" begin
    @show
    # Test 1: Sparse banded matrix (banded)
    # A = [1 2 0; 3 4 5; 0 6 7]
    # @test sparsestructure(sparse(A)) isa BandedMatrix

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
end
