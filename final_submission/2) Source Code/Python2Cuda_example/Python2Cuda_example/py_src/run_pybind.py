import sys
sys.path.append("../build/src/pybind11_cpp_examples/release")
sys.path.append("../build/src/pybind11_cuda_examples/release")
import cpp_add
import cpp_export
import cpp_matrix_add
import cu_matrix_add

import numpy as np
# Pyton inputs to C++
d = cpp_add.add(5, 7)
print(d)
e = cpp_add.subtract(5,7)
print(e)
# export variables from C++ to Python
print(cpp_export.value)
print(cpp_export.room_number)

# C++ class invoke
import cpp_class
pythonclass = cpp_class.Myclass("your name")
print(pythonclass.getname())
pythonclass.setname("my name is")
print(pythonclass.getname())

A = np.random.randint(10, size=(2,3))
B = np.random.randint(10, size=(2,3))

C = cpp_matrix_add.madd(A, B)
print(A)
print(B)
print(C)

C = cu_matrix_add.madd(A, B)
print(C)