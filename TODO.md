Possible things to implement:
 - Currently conv interpolation works only when grid has same number of spatial dimensions as channels. Other situations could also be implemented as simple broadcasting operation.
 - Sampling Jacobians of arbitrary mappings. The code base has been designed such that this would be relatively easy to implement.