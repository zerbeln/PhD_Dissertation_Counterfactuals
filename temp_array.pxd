# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "temp_array.hpp":
    cdef cppclass TempArray[T]:
        TempArray() except +
        TempArray(vector[T]*, size_t) except +
        TempArray(TempArray&) except +
        T& operator[](size_t) except +
        vector[T].iterator begin() except +
        vector[T].iterator end() except +
        