
## API reference

API reference has been generated using [pdoc](https://pdoc.dev/) by running the following command at directory `src`:

    pdoc composable_mapping -o ../docs --docformat google

## TODO
 - Currently conv interpolation works only when grid has same number of spatial dimensions as channels. Other situations could also be implemented as simple broadcasting operation.
 - Sampling Jacobians of arbitrary mappings. The code base has been designed such that this would be relatively easy to implement.
   - One could implement a generic implementation using torch.func.vjp.
   - For SamplableVolume one could revert to the generic implementation if derivative of the sampler is not implemented (that would require caching for mappable tensors to avoid unnecessary load from extra generate -calls).
 - Separate jacobian part and translation part in the affine transformation (this would e.g. allow composition of transformation with no translation easily and more efficiently).