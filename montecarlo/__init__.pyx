# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, infer_types=True,  embedsignature=True
ctypedef enum errors:
    ERR_PROB_ZERO,
    ERR_EMPTY_ARRAY

cdef double _montecarlo_rank(const double[::1] my_array,Py_ssize_t size_of_array,double target)nogil:
    """
    my_array: probabilities sorted in DESCENDING order
    target: p(alpha)
    returns: (1/n) * sum_{i: A[i] > target} 1/A[i]
    """
    if size_of_array == 0:
        with gil:raise ValueError("The size_of_array must be greater than 0")

    cdef double my_sum = 0
    cdef double prob = 0 
    for i in range(size_of_array):
        prob = my_array[i];
        if prob == 0.0:
            return 0.0
        if(prob<=target):
            break
        my_sum+= 1.0/prob

    return my_sum / size_of_array


cdef inline double bin_search(const double[::1]  my_array,Py_ssize_t size_of_array,double target)nogil:
    """
    my_array: probabilities sorted in DESCENDING order
    target: p(alpha)
    returns: (1/n) * sum_{i: A[i] > target} 1/A[i]
    """
    if size_of_array == 0:
        with gil:raise ValueError("The size_of_array must be greater than 0")

    cdef Py_ssize_t low = 0
    cdef Py_ssize_t mid
    cdef Py_ssize_t high = size_of_array

    while low<high:
        mid = (low+high)>>1
        if my_array[mid] > target:
            low = mid + 1 
        else:
            high = mid

    return low

cpdef double estimate_rank(probability:list,alpha:float) except * :
    cdef bool is_fast = True
    probability.sort(reverse=True)
    with nogil:
     if is_fast:
        result = bin_search(probability,len(probability),alpha)
     else:
        result = _montecarlo_rank(probability,len(probability),alpha)
    if result = -1.0:
        raise ValueError("There ")

