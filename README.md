# Composable mapping

*Composable mapping* is a PyTorch utility library developed for handling
coordinate mappings between images (2D or 3D), develped as part of SITReg, a
deep learning intra-modality image registration arhitecture fulfilling strict
symmetry properties.

Developed originally for medical imaging, this library provides a set of classes
and functions for handling spatial coordinate transformations.

The most powerful feature of this library is the ability to easily compose
transformations lazily and resample them to different coordinate systems as well
as sampler classes for sampling volumes defined on regular grids such that the
optimal method (either convolution or torch.grid_sample) is used based on the
sampling locations.

The main idea was to develop a library that allows handling of the coordinate
mappings as if they were mathematical functions, without losing much performance
compared to more manual implementation.

## Installation

Install using pip by running the command

    pip install git+https://github.com/honkamj/composable-mapping

## Requirements

- `Python 3.8+`
- `PyTorch 2.0+`

## Documentation

For a quick start tutorial, see [quick_start.ipynb](tutorials/quick_start.ipynb). For API reference, go to [https://honkamj.github.io/composable-mapping/](https://honkamj.github.io/composable-mapping/).

## SITReg

For SITReg implementation, see repository [SITReg](https://github.com/honkamj/SITReg).

## Publication

If you use composable mapping, please cite (see [bibtex](citations.bib)):

- **SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration using deformation inversion layers**  
[Joel Honkamaa](https://github.com/honkamj), Pekka Marttinen  
[eprint arXiv:2303.10211](https://arxiv.org/abs/2303.10211)

## License

Composable mapping is released under the MIT license.
