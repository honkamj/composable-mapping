"""Ceil integer division"""

from typing import Union


def ceildiv(denominator: Union[int, float], numerator: Union[int, float]) -> Union[int, float]:
    """Ceil integer division"""
    return -(denominator // -numerator)
