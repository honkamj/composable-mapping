"""Ceil integer division"""


def ceildiv(denominator: int, numerator: int) -> int:
    """Ceil integer division"""
    return -(denominator // -numerator)
