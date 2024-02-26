def hardmax(arr):
  """
  Performs the hardmax operation on an array, element-wise.

  Args:
      arr: A NumPy array of any shape.

  Returns:
      A NumPy array of the same shape as the input array, containing the index
      of the maximum element along each axis.
  """

  # Ensure the array is of type float for numerical stability
  arr = arr.astype(np.float32)

  # Find the indices of the maximum elements along each axis
  return np.argmax(arr, axis=1)
