#Importing sparse matrix class from sparse_recommender.py
from sparse_recommender import SparseMatrix
import pytest

#Initializing test sparse matrix
testsparsematrix = SparseMatrix(3,3)
testsparsematrix.set_value(0, 1, 5)
testsparsematrix.set_value(1, 2, 3)
testsparsematrix.set_value(2, 2, 4)

# Testing the set method by giving the row or column with out passing value will result in fail
def testing_set_method_fail1():
    try:
        assert testsparsematrix.set_value(0, 1)
    except Exception as e:
        print("Please enter all the 3 arguments properly")

# Testing the set method by giving the row or column with out of matrix size will result in fail
def testing_set_method_fail2():
    assert testsparsematrix.set_value(3, 4, 2) == "Invalid row or column index"

# Testing the set method by passing the null value will result in fail
def testing_set_method_fail3():
    assert testsparsematrix.set_value(0, 2, None) == "Please enter a value other than None"

#Testing the set menthod by passing a string value will result in fail
def testing_set_method_fail4():
    assert testsparsematrix.set_value(2, 0, "Movie") == "Please enter a number instead of string value"

# Testing the set method by passing a value within the matrix size with proper value will result in pass
def testing_set_method_pass1():
    testsparsematrix.set_value(2, 0, 1)
    assert testsparsematrix.get_value(2, 0) == 1

# Testing the set method by passing a value to an existing value will result in pass
def testing_set_method_pass2():
    testsparsematrix.set_value(0, 1, 1)
    assert testsparsematrix.get_value(0, 1) == 1

# Testing the get method with wrong value and resulting in Fail.
def testing_get_method_fail1():
    try:
        assert testsparsematrix.get_value(0, 1) == 3
    except AssertionError as e:
        print("Please test with correct number")

# Testing the get method by giving out of matrix size and resulting in Fail.
def testing_get_method_fail2():
    assert testsparsematrix.get_value(4, 4) == "Invalid row or column index"

# Testing the get method by giving more or less than 2 arguments will result in fail
def testing_get_method_fail3():
    try:
        testsparsematrix.get_value(0, 0, 1)
    except Exception as e:
        print("Please try to get the value with passing proper arguments")

# Testing the get method with correct value and resulting in Pass.
def testing_get_method_pass1():
    testsparsematrix.set_value(0,1,5)
    assert testsparsematrix.get_value(0, 1) == 5

# Testing Multiply method by passing more or less user vector compare to sparse matrix column size will result in fail

def test_multiply_method_fail1():
    test_user_vector1 = [1, 1, 1, 1]
    assert testsparsematrix.multiply_with_vector(test_user_vector1) == "Vector length must match the number of columns in the matrix"

# Testing Multiply method by passing user vectors correctly which matches user vector size with sparse matrix column size will result in pass

def test_multiply_method_pass1():
    test_user_vector2 = [1, 1, 1]
    result = testsparsematrix.multiply_with_vector(test_user_vector2)
    assert result == [5, 3, 5]

# Testing Add method by passing 2 different sizes of matrices will result in fail
test_other_sparsematrix = SparseMatrix(3,4)
def test_add_method_fail1():
    addition_result = testsparsematrix.add_matrix(test_other_sparsematrix)
    assert addition_result == "Matrix dimensions must match for addition"

# Testing Add method by passing 2 different sizes of matrices will result in fail

# Testing the add method
def test_add_method_pass1():
    test_other_sparsematrix1 = SparseMatrix(3,3)
    test_other_sparsematrix1.set_value(0, 0, 1)
    test_other_sparsematrix1.set_value(1, 1, 2)
    testsparsematrix = SparseMatrix(3,3)
    addition_result1 = testsparsematrix.add_matrix(test_other_sparsematrix1)
    assert addition_result1.to_dense_matrix() == [[1,0,0],[0,2,0],[0,0,0]]

# Testing convert to dense matrix method
def test_to_dense_matrix():
    test_other_sparsematrix1 = SparseMatrix(3,3)
    test_other_sparsematrix1.set_value(0, 0, 1)
    test_other_sparsematrix1.set_value(1, 1, 2)
    assert test_other_sparsematrix1.to_dense_matrix() == [[1,0,0],[0,2,0],[0,0,0]]