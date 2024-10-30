"""Base classes for composable mapping"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from matplotlib.figure import Figure  # type: ignore
from torch import Tensor

from .mappable_tensor import (
    IAffineTransformation,
    MappableTensor,
    concatenate_mappable_tensors,
    mappable,
    stack_mappable_tensors,
)
from .sampler import DataFormat, ISampler, get_sampler
from .tensor_like import BaseTensorLikeWrapper, ITensorLike
from .visualization import visualize_as_grid, visualize_as_image

if TYPE_CHECKING:
    from .coordinate_system import CoordinateSystem
Number = Union[int, float]


def _as_grid_composable_mapping_if_needed(
    target_mapping: "ComposableMapping", sources: Iterable[Any]
) -> "ComposableMapping":
    for source_mapping in sources:
        if isinstance(source_mapping, GridComposableMapping):
            return GridComposableMappingDecorator(target_mapping, source_mapping.coordinate_system)
    return target_mapping


@overload
def _bivariate_arithmetic_operator_template(
    mapping: "GridComposableMapping", other: Union["ComposableMapping", Number, MappableTensor]
) -> "GridComposableMapping": ...
@overload
def _bivariate_arithmetic_operator_template(
    mapping: "ComposableMapping", other: "GridComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _bivariate_arithmetic_operator_template(
    mapping: "ComposableMapping", other: Union["ComposableMapping", Number, MappableTensor]
) -> "ComposableMapping": ...
def _bivariate_arithmetic_operator_template(  # type: ignore
    mapping: "ComposableMapping",  # pylint: disable=unused-argument
    other: Union["ComposableMapping", Number, MappableTensor],  # pylint: disable=unused-argument
) -> "ComposableMapping": ...


ComposableMappingT = TypeVar("ComposableMappingT", bound="ComposableMapping")


def _univariate_arithmetic_operator_template(  # type: ignore
    mapping: "ComposableMappingT",  # pylint: disable=unused-argument
) -> "ComposableMappingT": ...


T = TypeVar("T")


def _generate_bivariate_arithmetic_operator(
    operator: Callable[[MappableTensor, Any], MappableTensor],
    inverse_operator: Callable[[MappableTensor, Any], MappableTensor],
    _type_template: T,
) -> T:
    def _operator(
        mapping: "ComposableMapping",
        other: Union["ComposableMapping", Number, MappableTensor],
    ) -> "ComposableMapping":
        return _as_grid_composable_mapping_if_needed(
            _ArithmeticOperator(
                mapping, other, operator=operator, inverse_operator=inverse_operator
            ),
            [mapping, other],
        )

    return cast(T, _operator)


def _generate_univariate_arithmetic_operator(
    operator: Callable[[MappableTensor], MappableTensor],
    inverse_operator: Callable[[MappableTensor], MappableTensor],
    _type_template: T,
) -> T:
    def _operator(mapping: "ComposableMapping") -> "ComposableMapping":
        return _as_grid_composable_mapping_if_needed(
            _ArithmeticOperator(
                mapping,
                None,
                operator=lambda x, _: operator(x),
                inverse_operator=lambda x, _: inverse_operator(x),
            ),
            [mapping],
        )

    return cast(T, _operator)


@overload
def _composition(
    left_mapping: "GridComposableMapping", right_mapping: "ComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _composition(
    left_mapping: "ComposableMapping", right_mapping: "GridComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _composition(
    left_mapping: "ComposableMapping", right_mapping: "ComposableMapping"
) -> "ComposableMapping": ...
def _composition(
    left_mapping: "ComposableMapping", right_mapping: "ComposableMapping"
) -> "ComposableMapping":
    return _as_grid_composable_mapping_if_needed(
        _Composition(left_mapping, right_mapping), [left_mapping, right_mapping]
    )


class ICoordinateSystemContainer(ABC):
    """Class holding a unique voxel coordinate system"""

    @property
    @abstractmethod
    def coordinate_system(
        self,
    ) -> "CoordinateSystem":
        """Get voxel coordinate system of the container"""


class ComposableMapping(ITensorLike, ABC):
    """Base class for composable mappings"""

    @abstractmethod
    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        """Evaluate the mapping at coordinates"""

    @abstractmethod
    def invert(self, **arguments) -> "ComposableMapping":
        """Invert the mapping

        Args:
            inversion_parameters: Possible inversion parameters
        """

    def sample_to(
        self,
        target: ICoordinateSystemContainer,
        *,
        data_format: Optional[DataFormat] = None,
    ) -> MappableTensor:
        """Sample the mapping with respect to the target coordinates"""
        if data_format is None:
            data_format = DataFormat()
        sampled = self(target.coordinate_system.grid())
        if data_format.representation == "displacements":
            sampled = sampled - target.coordinate_system.grid()
        if data_format.coordinate_type == "voxel":
            sampled = target.coordinate_system.to_voxel_coordinates(sampled)
        return sampled

    def resample_to(
        self,
        target: ICoordinateSystemContainer,
        *,
        data_format: Optional[DataFormat] = None,
        sampler: Optional["ISampler"] = None,
    ) -> "SamplableVolume":
        """Resample the deformation to the target coordinate system"""
        return SamplableVolume(
            data=self.sample_to(
                target,
                data_format=data_format,
            ),
            coordinate_system=target.coordinate_system,
            data_format=data_format,
            sampler=sampler,
        )

    def to_coordinates(self, target: "ICoordinateSystemContainer") -> "GridComposableMapping":
        """Set a coordinate system for the mapping, no resampling is done"""
        return GridComposableMappingDecorator(self, target.coordinate_system)

    def as_affine_transformation(self) -> IAffineTransformation:
        """Obtain the mapping as an affine transformation, if possible"""
        tracer = _AffineTracer()
        traced = self(tracer)
        if isinstance(traced, _AffineTracer):
            if traced.traced_affine is None:
                raise NotAffineTransformationError("Could not infer affine transformation")
            return traced.traced_affine
        raise NotAffineTransformationError("Could not infer affine transformation")

    def as_affine_matrix(self) -> Tensor:
        """Obtain the mapping as an affine matrix, if possible"""
        return self.as_affine_transformation().as_matrix()

    def visualize_to_as_deformed_grid(
        self,
        target: ICoordinateSystemContainer,
        *,
        batch_index: int = 0,
        figure_height: Number = 5,
        emphasize_every_nth_line: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """Visualize the mapping as a deformation

        If there are more than two dimension, central slices are shown for each pair of dimensions.

        Args:
            batch_index: Index of the batch element to visualize
            figure_height: Height of the figure
            emphasize_every_nth_line: Tuple of two integers, the first one is the number of lines
                to emphasize, the second one is the offset
        """
        return visualize_as_grid(
            self.sample_to(target),
            batch_index=batch_index,
            figure_height=figure_height,
            emphasize_every_nth_line=emphasize_every_nth_line,
        )

    def visualize_to_as_image(
        self,
        target: ICoordinateSystemContainer,
        batch_index: int = 0,
        figure_height: Number = 5,
        multiply_by_mask: bool = False,
        imshow_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Figure:
        """Visualize the mapping as an image"""
        return visualize_as_image(
            self.sample_to(target),
            voxel_size=target.coordinate_system.grid_spacing_cpu(),
            batch_index=batch_index,
            figure_height=figure_height,
            multiply_by_mask=multiply_by_mask,
            imshow_kwargs=imshow_kwargs,
        )

    __matmul__ = _composition
    __add__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x + y, lambda x, y: x - y, _bivariate_arithmetic_operator_template
    )
    __sub__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x - y, lambda x, y: x + y, _bivariate_arithmetic_operator_template
    )
    __mul__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x * y, lambda x, y: x / y, _bivariate_arithmetic_operator_template
    )
    __truediv__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x / y, lambda x, y: x * y, _bivariate_arithmetic_operator_template
    )
    __rtruediv__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: y / x, lambda x, y: y / x, _bivariate_arithmetic_operator_template
    )
    __neg__ = _generate_univariate_arithmetic_operator(
        lambda x: -x, lambda x: -x, _univariate_arithmetic_operator_template
    )


class GridComposableMapping(ComposableMapping, ICoordinateSystemContainer, ABC):
    """Base class for composable mappings coupled with a coordinate system"""

    def sample(
        self,
        *,
        data_format: Optional[DataFormat] = None,
    ) -> MappableTensor:
        """Sample the mapping with respect to the contained coordinates"""
        return self.sample_to(self, data_format=data_format)

    def resample(
        self,
        *,
        data_format: Optional[DataFormat] = None,
        sampler: Optional[ISampler] = None,
    ) -> "SamplableVolume":
        """Resample the deformation with respect to the contained coordinates"""
        return self.resample_to(
            self,
            data_format=data_format,
            sampler=sampler,
        )

    def visualize_as_deformed_grid(
        self,
        *,
        batch_index: int = 0,
        figure_height: int = 5,
        emphasize_every_nth_line: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """Visualize the mapping as a deformation"""
        return self.visualize_to_as_deformed_grid(
            self,
            batch_index=batch_index,
            figure_height=figure_height,
            emphasize_every_nth_line=emphasize_every_nth_line,
        )

    def visualize_as_image(self, **kwargs) -> Figure:
        """Visualize the mapping as an image"""
        return self.visualize_to_as_image(self, **kwargs)


class GridComposableMappingDecorator(BaseTensorLikeWrapper, GridComposableMapping):
    """Base decorator for composable mappings"""

    def __init__(self, mapping: ComposableMapping, coordinate_system: "CoordinateSystem") -> None:
        self._mapping = mapping
        self._coordinate_system = coordinate_system

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"mapping": self._mapping, "coordinate_system": self._coordinate_system}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridComposableMappingDecorator":
        return GridComposableMappingDecorator(
            cast(ComposableMapping, children["mapping"]),
            cast("CoordinateSystem", children["coordinate_system"]),
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        return self._mapping(coordinates)

    def invert(self, **arguments) -> ComposableMapping:
        return GridComposableMappingDecorator(
            self._mapping.invert(**arguments), self._coordinate_system
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self._coordinate_system

    def __repr__(self) -> str:
        return (
            f"GridComposableMappingDecorator(mapping={self._mapping}, "
            f"coordinate_system={self._coordinate_system})"
        )


class Identity(BaseTensorLikeWrapper, ComposableMapping):
    """Identity mapping"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "Identity":
        return Identity()

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        return coordinates

    def invert(self, **_inversion_parameters) -> "Identity":
        return Identity()

    def detach(self) -> "Identity":
        return self

    def __repr__(self) -> str:
        return "Identity()"


class _Composition(BaseTensorLikeWrapper, ComposableMapping):
    """Composition of two mappings"""

    def __init__(self, left_mapping: ComposableMapping, right_mapping: ComposableMapping) -> None:
        self._left_mapping = left_mapping
        self._right_mapping = right_mapping

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_Composition":
        return _Composition(
            cast(ComposableMapping, children["left_mapping"]),
            cast(ComposableMapping, children["right_mapping"]),
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"left_mapping": self._left_mapping, "right_mapping": self._right_mapping}

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return self._left_mapping(self._right_mapping(masked_coordinates))

    def invert(self, **arguments) -> "ComposableMapping":
        return _Composition(
            self._right_mapping.invert(**arguments),
            self._left_mapping.invert(**arguments),
        )

    def __repr__(self) -> str:
        return (
            f"_Composition(left_mapping={self._left_mapping}, right_mapping={self._right_mapping})"
        )


class _ArithmeticOperator(BaseTensorLikeWrapper, ComposableMapping):
    """Aritmetic operator of a composable mapping"""

    def __init__(
        self,
        mapping: ComposableMapping,
        other: Optional[Union[ComposableMapping, MappableTensor, Number, Tensor]],
        operator: Callable[[MappableTensor, Any], MappableTensor],
        inverse_operator: Callable[[MappableTensor, Any], MappableTensor],
    ) -> None:
        self._mapping = mapping
        self._other = other
        self._operator = operator
        self._inverse_operator = inverse_operator

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_ArithmeticOperator":
        return _ArithmeticOperator(
            cast(ComposableMapping, self._mapping),
            cast(Union[ComposableMapping, MappableTensor, Number, Tensor], tensors["other"]),
            self._operator,
            self._inverse_operator,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        tensors = {}
        if isinstance(self._other, Tensor):
            tensors["other"] = self._other
        return tensors

    def _get_children(self) -> Mapping[str, ITensorLike]:
        children: Dict[str, ITensorLike] = {"mapping": self._mapping}
        if isinstance(self._other, MappableTensor) or isinstance(self._other, ComposableMapping):
            children["other"] = self._other
        return children

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        if isinstance(self._other, ComposableMapping):
            return self._operator(
                self._mapping(masked_coordinates), self._other(masked_coordinates)
            )
        return self._operator(self._mapping(masked_coordinates), self._other)

    def invert(self, **arguments) -> "ComposableMapping":
        if isinstance(self._other, ComposableMapping):
            raise ValueError("Operation is not invertible")
        return _Composition(
            self._mapping.invert(**arguments),
            _ArithmeticOperator(Identity(), self._other, self._inverse_operator, self._operator),
        )

    def __repr__(self) -> str:
        return (
            f"_ArithmeticOperator(mapping={self._mapping}, "
            "other={self._other}, operator={self._operator}, "
            "inverse_operator={self._inverse_operator})"
        )


class SamplableVolume(BaseTensorLikeWrapper, GridComposableMapping):
    """Continuously defined mapping based on regular grid samples

    Arguments:
        data: Regular grid of values defining the deformation, with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The grid should be
            in voxel coordinates.
        sampler: Sampler for the grid
    """

    def __init__(
        self,
        data: MappableTensor,
        coordinate_system: "CoordinateSystem",
        *,
        data_format: Optional[DataFormat] = None,
        sampler: Optional[ISampler] = None,
    ) -> None:
        self._data = data
        self._coordinate_system = coordinate_system
        self._data_format = DataFormat() if data_format is None else data_format
        self._sampler = get_sampler(sampler)

    def modify_sampler(self, sampler: ISampler) -> "SamplableVolume":
        """Modify the sampler of the volume"""
        return SamplableVolume(
            data=self._data,
            coordinate_system=self._coordinate_system,
            data_format=self._data_format,
            sampler=sampler,
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self._coordinate_system

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {
            "data": self._data,
            "coordinate_system": self._coordinate_system,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "SamplableVolume":
        return SamplableVolume(
            data=cast(MappableTensor, children["data"]),
            coordinate_system=cast("CoordinateSystem", children["coordinate_system"]),
            data_format=self._data_format,
            sampler=self._sampler,
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        voxel_coordinates = self._coordinate_system.to_voxel_coordinates(coordinates)
        sampled = self._sampler(self._data, voxel_coordinates)
        if self._data_format.representation == "displacements":
            if self._data_format.coordinate_type == "voxel":
                sampled = voxel_coordinates + sampled
            elif self._data_format.coordinate_type == "world":
                sampled = coordinates + sampled
        if self._data_format.coordinate_type == "voxel":
            sampled = self._coordinate_system.from_voxel_coordinates(sampled)
        return sampled

    def invert(self, **arguments) -> ComposableMapping:
        return SamplableVolume(
            data=self._data,
            coordinate_system=self._coordinate_system,
            data_format=self._data_format,
            sampler=self._sampler.inverse(self._coordinate_system, self._data_format, arguments),
        )

    def __repr__(self) -> str:
        return (
            f"SamplableVolume(data={self._data}, "
            f"coordinate_system={self._coordinate_system}, "
            f"data_format={self._data_format}, "
            f"sampler={self._sampler})"
        )


def samplable_volume(
    data: Tensor,
    coordinate_system: "CoordinateSystem",
    mask: Optional[Tensor] = None,
    *,
    data_format: Optional[DataFormat] = None,
    sampler: Optional[ISampler] = None,
) -> GridComposableMapping:
    """Create a continuously defined mapping based on regular grid samples"""
    return SamplableVolume(
        data=mappable(data, mask),
        coordinate_system=coordinate_system,
        data_format=data_format,
        sampler=sampler,
    )


class NotAffineTransformationError(Exception):
    """Error raised when a composable mapping is not affine"""


class _AffineTracer(MappableTensor):
    # pylint: disable=super-init-not-called
    def __init__(self, affine_transformation: Optional[IAffineTransformation] = None) -> None:
        self.traced_affine: Optional[IAffineTransformation] = affine_transformation

    def transform(self, affine_transformation: IAffineTransformation) -> MappableTensor:
        if self.traced_affine is not None:
            traced_affine = affine_transformation @ self.traced_affine
        else:
            traced_affine = affine_transformation
        return _AffineTracer(traced_affine)

    def __getattribute__(self, name: str):
        if name not in ("transform", "traced_affine"):
            raise NotAffineTransformationError(
                "Could not infer affine transformation since an other operation "
                f"than applying affine transformation was applied ({name})"
            )
        return object.__getattribute__(self, name)


class _Stack(BaseTensorLikeWrapper, ComposableMapping):
    """Stacked mappings"""

    def __init__(self, *mappings: ComposableMapping, channel_index: int) -> None:
        self._mappings = mappings
        self._channel_index = channel_index

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_Stack":
        return _Stack(
            *(cast(ComposableMapping, children[f"mapping_{i}"]) for i in range(len(children))),
            channel_index=self._channel_index,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        children = {}
        for i, mapping in enumerate(self._mappings):
            children[f"mapping_{i}"] = mapping
        return children

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return stack_mappable_tensors(
            *(mapping(masked_coordinates) for mapping in self._mappings),
            channel_index=self._channel_index,
        )

    def invert(self, **arguments) -> "ComposableMapping":
        raise NotImplementedError("Inversion of stacked mappings is not implemented")

    def __repr__(self) -> str:
        return f"_Stack(mappings={self._mappings}, " f"channel_index={self._channel_index})"


class _Concatenate(BaseTensorLikeWrapper, ComposableMapping):
    """Concatenated mappings"""

    def __init__(self, *mappings: ComposableMapping, channel_index: int) -> None:
        self._mappings = mappings
        self._channel_index = channel_index

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_Concatenate":
        return _Concatenate(
            *(cast(ComposableMapping, children[f"mapping_{i}"]) for i in range(len(children))),
            channel_index=self._channel_index,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        children = {}
        for i, mapping in enumerate(self._mappings):
            children[f"mapping_{i}"] = mapping
        return children

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return concatenate_mappable_tensors(
            *(mapping(masked_coordinates) for mapping in self._mappings),
            channel_index=self._channel_index,
        )

    def invert(self, **arguments) -> "ComposableMapping":
        raise NotImplementedError("Inversion of stacked mappings is not implemented")

    def __repr__(self) -> str:
        return f"_Concatenate(mappings={self._mappings}, " f"channel_index={self._channel_index})"


def stack_mappings(*mappings: ComposableMappingT, channel_index: int = 0) -> ComposableMappingT:
    """Stack mappings along the channel dimension"""
    stacked: ComposableMapping = _Stack(*mappings, channel_index=channel_index)
    for mapping in mappings:
        if isinstance(mapping, GridComposableMapping):
            stacked = GridComposableMappingDecorator(stacked, mapping.coordinate_system)
    return cast(ComposableMappingT, stacked)


def concatenate_mappings(
    *mappings: ComposableMappingT, channel_index: int = 0
) -> ComposableMappingT:
    """Concatenate mappings along the channel dimension"""
    concatenated: ComposableMapping = _Concatenate(*mappings, channel_index=channel_index)
    for mapping in mappings:
        if isinstance(mapping, GridComposableMapping):
            concatenated = GridComposableMappingDecorator(concatenated, mapping.coordinate_system)
    return cast(ComposableMappingT, concatenated)
