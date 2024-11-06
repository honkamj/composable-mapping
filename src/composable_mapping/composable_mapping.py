"""Composable mapping."""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from torch import Tensor

from .affine_transformation import IAffineTransformation
from .interface import Number
from .mappable_tensor import (
    MappableTensor,
    concatenate_mappable_tensors,
    mappable,
    stack_mappable_tensors,
)
from .sampler import DataFormat, ISampler, get_sampler
from .tensor_like import BaseTensorLikeWrapper, ITensorLike

if TYPE_CHECKING:
    from .coordinate_system import CoordinateSystem


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
            _BivariateArithmeticOperator(
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
            _UnivariateArithmeticOperator(
                mapping,
                operator=operator,
                inverse_operator=inverse_operator,
            ),
            [mapping],
        )

    return cast(T, _operator)


@overload
def _composition(
    self: "GridComposableMapping", right_mapping: "ComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _composition(
    self: "ComposableMapping", right_mapping: "GridComposableMapping"
) -> "GridComposableMapping": ...
@overload
def _composition(
    self: "ComposableMapping", right_mapping: "ComposableMapping"
) -> "ComposableMapping": ...
def _composition(
    self: "ComposableMapping", right_mapping: "ComposableMapping"
) -> "ComposableMapping":
    return _as_grid_composable_mapping_if_needed(
        _Composition(self, right_mapping), [self, right_mapping]
    )


class ICoordinateSystemContainer(ABC):
    """Class holding a unique coordinate system."""

    @property
    @abstractmethod
    def coordinate_system(
        self,
    ) -> "CoordinateSystem":
        """Coordinate system of the container."""


class ComposableMapping(ITensorLike, ABC):
    """Base class for mappings composable with each other and action on mappable
    tensors.

    In general a composable mapping is a callable object that takes coordinates
    as input and returns the mapping evaluated at these coordinates. Composable
    mappings are generally assumed to be independent over the batch and spatial
    dimensions of an input.

    As the name suggests, a composable mapping can be additionally composed with
    other composable mappings. Basic arithmetic operations are also supported
    between two composable mappings or between a composable mapping and a number
    or a tensor, both of which return a new composable mapping.
    """

    @abstractmethod
    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        """Evaluate the mapping at coordinates.

        Args:
            coordinates: Coordinates to evaluate the mapping at with shape
                ([*batch_shape ,]n_dims[, *spatial_shape]).

        Returns:
            Mapping evaluated at the coordinates.

        @public
        """

    @abstractmethod
    def invert(self, **arguments) -> "ComposableMapping":
        """Invert the mapping.

        Args:
            arguments: Arguments for the inversion.

        Returns:
            The inverted mapping.
        """

    def sample_to(
        self,
        target: ICoordinateSystemContainer,
        data_format: Optional[DataFormat] = None,
    ) -> MappableTensor:
        """Evaluate the mapping at the coordinates defined by the target.

        Args:
            target: Target coordinate system (or a container with a coordinate system)
                defining a grid to evaluate the mapping at.
            data_format: Data format of the output. Default data format depends
                on the mapping, and can be accessed through the property
                `default_sampling_data_format`.

        Returns:
            Mappable tensor containing the values obtained by evaluating the
            mapping at the coordinates defined by the target.
        """
        data_format = self._get_sampling_data_format(data_format)
        sampled = self(target.coordinate_system.grid())
        if data_format.coordinate_type == "voxel":
            sampled = target.coordinate_system.to_voxel_coordinates(sampled)
        if data_format.representation == "displacements":
            grid = (
                target.coordinate_system.voxel_grid()
                if data_format.coordinate_type == "voxel"
                else target.coordinate_system.grid()
            )
            sampled = sampled - grid
        return sampled

    def resample_to(
        self,
        target: ICoordinateSystemContainer,
        data_format: Optional[DataFormat] = None,
        sampler: Optional["ISampler"] = None,
    ) -> "SamplableVolume":
        """Resample the mapping at the coordinates defined by the target.

        Args:
            target: Target coordinate system (or a container with a coordinate system)
                defining a grid to resample the mapping at.
            data_format: Data format used as an internal representation of the
                generated resampled mapping. Default data format depends
                on the mapping, and can be accessed through the property
                `default_sampling_data_format`.
            sampler: Sampler used by the generated resampled mapping. Note that
                this sampler is not used to resample the mapping, but to sample
                the generated resampled mapping. If None, the default sampler
                is used (see `default_sampler`).

        Returns:
            Resampled mapping.
        """
        data_format = self._get_sampling_data_format(data_format)
        return SamplableVolume(
            data=self.sample_to(
                target,
                data_format=data_format,
            ),
            coordinate_system=target.coordinate_system,
            data_format=data_format,
            sampler=sampler,
        )

    def assign_coordinates(
        self, coordinates: "ICoordinateSystemContainer"
    ) -> "GridComposableMapping":
        """Assign a coordinate system for the mapping.

        This only changes the coordinate system of the mapping, the mapping itself
        is not changed. The coordinate system contained by the mapping affects
        behaviour of some methods such as `GridComposableMapping.sample` and
        `GridComposableMapping.resample`.

        Args:
            coordinates: Coordinate system (or a container with a coordinate system)
                to assign for the mapping.

        Returns:
            Mapping with the given target coordinate system.
        """
        return GridComposableMappingDecorator(self, coordinates.coordinate_system)

    def as_affine_transformation(self) -> IAffineTransformation:
        """Obtain the mapping as an affine transformation on PyTorch tensors, if possible.

        Returns:
            Affine transformation on PyTorch tensors.

        Raises:
            NotAffineTransformationError: If the mapping is not an affine transformation on
                PyTorch tensors.
        """
        tracer = _AffineTracer()
        traced = self(tracer)
        if isinstance(traced, _AffineTracer):
            if traced.traced_affine is None:
                raise NotAffineTransformationError("Could not infer affine transformation")
            return traced.traced_affine
        raise NotAffineTransformationError("Could not infer affine transformation")

    def as_affine_matrix(self) -> Tensor:
        """Obtain the mapping as an affine matrix, if possible.

        Returns:
            Affine matrix with
            shape ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
        """
        return self.as_affine_transformation().as_matrix()

    @property
    def default_sampling_data_format(self) -> Optional[DataFormat]:
        """Default data format to use in sampling and resampling operations for
        the mapping.

        If None, DataFormat.world_coordinates() will be used but the behaviour
        in operations with other mappings is different as the default data format
        of the other mapping will be used.
        """
        return None

    def _get_sampling_data_format(self, data_format: Optional[DataFormat]) -> DataFormat:
        if data_format is not None:
            return data_format
        if self.default_sampling_data_format is None:
            return DataFormat.world_coordinates()
        return self.default_sampling_data_format

    __matmul__ = _composition
    """Compose with another mapping.
    
    Args:
        right_mapping: Mapping to compose with.
    
    Returns:
        Composed mapping.
    @public
    """
    __add__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x + y, lambda x, y: x - y, _bivariate_arithmetic_operator_template
    )
    __radd__ = __add__
    __sub__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x - y, lambda x, y: x + y, _bivariate_arithmetic_operator_template
    )
    __rsub__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: y - x, lambda x, y: y - x, _bivariate_arithmetic_operator_template
    )
    __mul__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x * y, lambda x, y: x / y, _bivariate_arithmetic_operator_template
    )
    __rmul__ = __mul__
    __truediv__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x / y, lambda x, y: x * y, _bivariate_arithmetic_operator_template
    )
    __rtruediv__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: y / x, lambda x, y: y / x, _bivariate_arithmetic_operator_template
    )
    __pow__ = _generate_bivariate_arithmetic_operator(
        lambda x, y: x**y, lambda x, y: x ** (1 / y), _bivariate_arithmetic_operator_template
    )
    __neg__ = _generate_univariate_arithmetic_operator(
        lambda x: -x, lambda x: -x, _univariate_arithmetic_operator_template
    )


class GridComposableMapping(ComposableMapping, ICoordinateSystemContainer, ABC):
    """Base class for composable mappings coupled with a coordinate system."""

    @abstractmethod
    def invert(self, **arguments) -> "GridComposableMapping":
        pass

    def sample(
        self,
        data_format: Optional[DataFormat] = None,
    ) -> MappableTensor:
        """Evaluate the mapping at the coordinates contained by the mapping.

        Args:
            data_format: Data format of the output. Default data format depends
                on the mapping, and can be accessed through the property
                `default_sampling_data_format`.

        Returns:
            Mappable tensor containing the values obtained by evaluating the
            mapping at the coordinates contained by the mapping.
        """
        return self.sample_to(self, data_format=data_format)

    def resample(
        self,
        data_format: Optional[DataFormat] = None,
        sampler: Optional[ISampler] = None,
    ) -> "SamplableVolume":
        """Resample the mapping at the coordinates contained by the mapping.

        Args:
            data_format: Data format used as an internal representation of the
                generated resampled mapping. Default data format depends
                on the mapping, and can be accessed through the property
                `default_sampling_data_format`.
            sampler: Sampler used by the generated resampled mapping. Note that
                this sampler is not used to resample the mapping, but to sample
                the generated resampled mapping. If None, the default sampler
                is used (see `default_sampler`).

        Returns:
            Resampled mapping.
        """
        return self.resample_to(
            self,
            data_format=data_format,
            sampler=sampler,
        )


class GridComposableMappingDecorator(BaseTensorLikeWrapper, GridComposableMapping):
    """Decorator for coupling a composable mapping with a coordinate system.

    Args:
        mapping: Composable mapping.
        coordinate_system: Coordinate system assigned to the mapping.

    @private
    """

    def __init__(self, mapping: ComposableMapping, coordinate_system: "CoordinateSystem") -> None:
        super().__init__()
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

    def invert(self, **arguments) -> GridComposableMapping:
        return GridComposableMappingDecorator(
            self._mapping.invert(**arguments), self._coordinate_system
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self._coordinate_system

    @property
    def default_sampling_data_format(self) -> Optional[DataFormat]:
        return self._mapping.default_sampling_data_format

    def __repr__(self) -> str:
        return (
            f"GridComposableMappingDecorator(mapping={self._mapping}, "
            f"coordinate_system={self._coordinate_system})"
        )


class Identity(BaseTensorLikeWrapper, ComposableMapping):
    """Identity mapping."""

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
    """Composition of two mappings."""

    def __init__(self, left_mapping: ComposableMapping, right_mapping: ComposableMapping) -> None:
        super().__init__()
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

    @property
    def default_sampling_data_format(self) -> Optional[DataFormat]:
        if self._left_mapping.default_sampling_data_format is not None:
            return self._left_mapping.default_sampling_data_format
        return self._right_mapping.default_sampling_data_format

    def __repr__(self) -> str:
        return (
            f"_Composition(left_mapping={self._left_mapping}, right_mapping={self._right_mapping})"
        )


class _BivariateArithmeticOperator(BaseTensorLikeWrapper, ComposableMapping):
    def __init__(
        self,
        mapping: ComposableMapping,
        other: Union[ComposableMapping, MappableTensor, Number, Tensor],
        operator: Callable[[MappableTensor, Any], MappableTensor],
        inverse_operator: Callable[[MappableTensor, Any], MappableTensor],
    ) -> None:
        super().__init__()
        self._mapping = mapping
        self._other = other
        self._operator = operator
        self._inverse_operator = inverse_operator

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_BivariateArithmeticOperator":
        return _BivariateArithmeticOperator(
            self._mapping,
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
        if isinstance(self._other, (MappableTensor, ComposableMapping)):
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
            _BivariateArithmeticOperator(
                Identity(), self._other, self._inverse_operator, self._operator
            ),
        )

    @property
    def default_sampling_data_format(self) -> Optional[DataFormat]:
        if self._mapping.default_sampling_data_format is not None:
            return self._mapping.default_sampling_data_format
        if isinstance(self._other, ComposableMapping):
            return self._other.default_sampling_data_format
        return None

    def __repr__(self) -> str:
        return (
            f"_BivariateArithmeticOperator(mapping={self._mapping}, "
            "other={self._other}, operator={self._operator}, "
            "inverse_operator={self._inverse_operator})"
        )


class _UnivariateArithmeticOperator(BaseTensorLikeWrapper, ComposableMapping):
    def __init__(
        self,
        mapping: ComposableMapping,
        operator: Callable[[MappableTensor], MappableTensor],
        inverse_operator: Callable[[MappableTensor], MappableTensor],
    ) -> None:
        super().__init__()
        self._mapping = mapping
        self._operator = operator
        self._inverse_operator = inverse_operator

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_UnivariateArithmeticOperator":
        return _UnivariateArithmeticOperator(
            cast(ComposableMapping, self._mapping),
            self._operator,
            self._inverse_operator,
        )

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"mapping": self._mapping}

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return self._operator(self._mapping(masked_coordinates))

    def invert(self, **arguments) -> "ComposableMapping":
        return _Composition(
            self._mapping.invert(**arguments),
            _UnivariateArithmeticOperator(Identity(), self._inverse_operator, self._operator),
        )

    @property
    def default_sampling_data_format(self) -> Optional[DataFormat]:
        return self._mapping.default_sampling_data_format

    def __repr__(self) -> str:
        return (
            f"_UnivariateArithmeticOperator(mapping={self._mapping}, "
            "operator={self._operator}, "
            "inverse_operator={self._inverse_operator})"
        )


class SamplableVolume(BaseTensorLikeWrapper, GridComposableMapping):
    """Mapping defined based on a regular grid of values and a sampler turning the
    grid values into a continuously defined mapping.

    The easiest way to create a samplable volume is to use the factory
    function provided in this module or the class method of this class:
    `samplable_volume`, `SamplableVolume.from_tensor`.

    Arguments:
        data: Regular grid of values, with shape
            (*batch_shape, *channels_shape, *spatial_shape).
        coordinate_system: Coordinate system describing transformation from the
            voxel coordinates on the data grid to the world coordinates.
        data_format: Data format of the grid values.
        sampler: Sampler turning the grid values into a continuously defined mapping
            over spatial coordinates.
    """

    def __init__(
        self,
        data: MappableTensor,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat = DataFormat.world_coordinates(),
        sampler: Optional[ISampler] = None,
    ) -> None:
        super().__init__()
        if coordinate_system.spatial_shape != data.spatial_shape:
            raise ValueError(
                "Coordinate system spatial shape must match the data spatial shape. "
                f"Coordinate system spatial shape: {coordinate_system.spatial_shape}, "
                f"data spatial shape: {data.spatial_shape}."
            )
        self._data = data
        self._coordinate_system = coordinate_system
        self._data_format = data_format
        self._sampler = get_sampler(sampler)

    def from_tensor(
        self,
        data: Tensor,
        coordinate_system: "CoordinateSystem",
        mask: Optional[Tensor] = None,
        data_format: DataFormat = DataFormat.world_coordinates(),
        sampler: Optional[ISampler] = None,
    ):
        """Create a samplable volume from a tensor.

        Args:
            data: Regular grid of values, with shape
                (*batch_shape, *channels_shape, *spatial_shape).
            coordinate_system: Coordinate system describing transformation from the
                voxel coordinates on the data grid to the world coordinates.
            mask: Mask for the data,
                with shape (*batch_shape, *(1,) * n_channel_dims, *spatial_shape).
            data_format: Data format of the grid values.
            sampler: Sampler turning the grid values into a continuously defined mapping
                over spatial coordinates.

        Returns:
            Samplable volume.
        """
        return SamplableVolume(
            data=mappable(data, mask),
            coordinate_system=coordinate_system,
            data_format=data_format,
            sampler=sampler,
        )

    def modify_sampler(self, sampler: ISampler) -> "SamplableVolume":
        """Modify the sampler of the volume.

        Args:
            sampler: New sampler.

        Returns:
            Samplable volume with the new sampler.
        """
        return SamplableVolume(
            data=self._data,
            coordinate_system=self._coordinate_system,
            data_format=self._data_format,
            sampler=sampler,
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self._coordinate_system

    @property
    def default_sampling_data_format(self) -> DataFormat:
        return self._data_format

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

    def invert(self, **arguments) -> "SamplableVolume":
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
    data_format: DataFormat = DataFormat.world_coordinates(),
    sampler: Optional[ISampler] = None,
) -> GridComposableMapping:
    """Create a samplable volume from a tensor.

    See: `SamplableVolume.from_tensor`.
    """
    return SamplableVolume(
        data=mappable(data, mask),
        coordinate_system=coordinate_system,
        data_format=data_format,
        sampler=sampler,
    )


class NotAffineTransformationError(Exception):
    """Error raised when trying to represent a non-affine composable mapping as
    an affine transformation."""


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
    """Stacked mappings."""

    def __init__(self, *mappings: ComposableMapping, channel_index: int) -> None:
        super().__init__()
        self._mappings = mappings
        self._channel_index = channel_index

    @property
    def default_sampling_data_format(self) -> DataFormat:
        return DataFormat.world_coordinates()

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
    """Concatenated mappings."""

    def __init__(self, *mappings: ComposableMapping, channel_index: int) -> None:
        super().__init__()
        self._mappings = mappings
        self._channel_index = channel_index

    @property
    def default_sampling_data_format(self) -> DataFormat:
        return DataFormat.world_coordinates()

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
    """Stack mappings along a channel dimension.

    Args:
        mappings: Mappings to stack.
        channel_index: Channel index along which to stack.

    Returns:
        A mapping with the output being the outputs of the input mappings
        stacked along the channel dimension.
    """
    stacked: ComposableMapping = _Stack(*mappings, channel_index=channel_index)
    for mapping in mappings:
        if isinstance(mapping, GridComposableMapping):
            stacked = GridComposableMappingDecorator(stacked, mapping.coordinate_system)
    return cast(ComposableMappingT, stacked)


def concatenate_mappings(
    *mappings: ComposableMappingT, channel_index: int = 0
) -> ComposableMappingT:
    """Concatenate mappings along a channel dimension.

    Args:
        mappings: Mappings to concatenate.
        channel_index: Channel index along which to concatenate.

    Returns:
        A mapping with the output being the outputs of the input mappings
        concatenated along the channel dimension.
    """
    concatenated: ComposableMapping = _Concatenate(*mappings, channel_index=channel_index)
    for mapping in mappings:
        if isinstance(mapping, GridComposableMapping):
            concatenated = GridComposableMappingDecorator(concatenated, mapping.coordinate_system)
    return cast(ComposableMappingT, concatenated)
