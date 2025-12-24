import numpy as np
import cuda_bind_py

# Generate random input matrices
input1 = np.random.randint(0, 10, (1024, 1024), dtype=np.int32)
input2 = np.random.randint(0, 10, (1024, 1024), dtype=np.int32)

# Call the CUDA function
output = cuda_bind_py.matrix_multiply(input1, input2)

print("Matrix multiplication completed!")
print("Output[0,0] =", output[0,0])