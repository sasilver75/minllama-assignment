import torch
from torch import tensor

# Create a 2x3 tensor with some example values
example_tensor = tensor([
    # batch element 0
    [
        [ 1.0,  2.0,  3.0,  4.0],   # token 0 embedding
        [ 5.0,  6.0,  7.0,  8.0],   # token 1 embedding
        [ 9.0, 10.0, 11.0, 12.0]    # token 2 embedding
    ],
    # batch element 1
    [
        [13.0, 14.0, 15.0, 16.0],   # token 0 embedding
        [17.0, 18.0, 19.0, 20.0],   # token 1 embedding
        [21.0, 22.0, 23.0, 24.0]    # token 2 embedding
    ],
])

# Now let's apply RMSNorm to this tensor in steps!

# First, we compute the square of the tensor
print(f"Example tensor shape: {example_tensor.shape}")


# Now, let's perform RMSNorm in steps!



# For each embedding vector in our batch, compute the mean of its squared elements.
# This is along dimension 2, which is the embedding dimension.
mean_squared_elements = torch.mean(example_tensor ** 2, dim=2, keepdim=True)

print(f"Mean squared elements shape: {mean_squared_elements.shape}") # (2,3)

eps = 1e-5

# We can add an epsilon for stability, inside the SQRT
mean_squared_elements_and_eps = mean_squared_elements + eps

print(f"Mean squared elements and epsilon shape: {mean_squared_elements_and_eps.shape}") # (2,3)

# Now, take the square root of the mean squared elements
sqrt_mean_squared_elements = torch.sqrt(mean_squared_elements_and_eps)

print(f"Square root of mean squared elements shape: {sqrt_mean_squared_elements.shape}")

# Then we divide the original tensor by this sqrt tensor
normalized_tensor = example_tensor / sqrt_mean_squared_elements

# And then there's a final scaling gain vector gamma, which is a learnable parameter
gamma = torch.nn.Parameter(torch.ones(1, 1, 4))

# So we multiply the normalized tensor by gamma
scaled_tensor = normalized_tensor * gamma

print(f"Scaled tensor shape: {scaled_tensor.shape}")
print(scaled_tensor)


x = torch.tensor([1,2,3])
print(x / 3)
print(x.shape[-1])

