# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Periodic Cartesian constraints and projected flow on hexahedral meshes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import la, mesh
from dolfinx.fem import Form, Function, FunctionSpace, assemble_matrix, assemble_vector, form
from dolfinx.fem.function import functionspace

if TYPE_CHECKING:
    import cupy as cp
    from scipy.sparse import csr_matrix


__all__ = ["PeriodicDofMap", "PeriodicIncompressibleProblem"]


class PeriodicDofMap:
    """Identify opposite boundaries of an equispaced nodal Cartesian grid.

    The reduced indices follow Cartesian order, with components contiguous
    at each node. The original space includes both endpoints of each axis.
    This map is for assembled coefficient and test vectors on one MPI rank.
    """

    def __init__(
        self,
        space: FunctionSpace,
        shape: tuple[int, int, int],
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
    ) -> None:
        """Build a map for the given nodal space and periodic grid.

        Args:
            space: Complete nodal space on a serial, three-dimensional mesh.
            shape: Number of distinct periodic nodes along each axis.
            lower: Coordinates of the lower corner of the grid.
            upper: Coordinates of the opposite corner of the grid.
        """
        shape = tuple(shape)
        extent = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
        if (
            len(shape) != 3
            or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape)
            or extent.shape != (3,)
            or np.any(~np.isfinite(extent))
            or np.any(extent <= 0)
        ):
            raise ValueError("Provide a positive three-dimensional shape and finite grid bounds.")
        if space.mesh.comm.size != 1 or space.mesh.geometry.dim != 3:
            raise ValueError(
                "Periodic DOF identification requires a serial three-dimensional mesh."
            )
        if len(space.dofmaps) != 1 or space.element.needs_dof_transformations:
            raise ValueError("Periodic DOF identification requires a nodal space with one dofmap.")

        self.shape = shape
        self.block_size = space.dofmap.index_map_bs
        self.size = math.prod(shape) * self.block_size
        coordinates = space.tabulate_dof_coordinates()
        scaled = (coordinates - lower) * (np.asarray(shape) / extent)
        rounded = np.rint(scaled)
        if not np.allclose(scaled, rounded, rtol=0, atol=1e-9):
            raise ValueError("Expected nodal coordinates on the requested Cartesian lattice.")
        lattice = rounded.astype(np.int64)
        if np.any(lattice < 0) or np.any(lattice > shape):
            raise ValueError("Nodal coordinates extend outside the periodic box.")

        full_shape = tuple(n + 1 for n in shape)
        full = np.ravel_multi_index(lattice.T, full_shape)
        if len(coordinates) != math.prod(full_shape) or np.unique(full).size != len(coordinates):
            raise ValueError("Expected exactly one original node at every Cartesian grid point.")
        if space.dofmap.index_map.size_local != len(coordinates):
            raise ValueError("Expected one DOF block at each Cartesian grid point.")

        self.node_indices = np.ravel_multi_index((lattice % shape).T, shape)
        self.indices = (
            self.block_size * self.node_indices[:, None] + np.arange(self.block_size)
        ).ravel()
        canonical = np.flatnonzero(np.all(lattice < shape, axis=1))
        self._canonical_nodes = np.empty(math.prod(shape), dtype=np.intp)
        self._canonical_nodes[self.node_indices[canonical]] = canonical

    def restrict(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Read each periodic coefficient from the grid with the upper endpoints omitted."""
        if values.shape != self.indices.shape:
            raise ValueError("Expected a flat vector in the original DOF layout.")
        return values.reshape(-1, self.block_size)[self._canonical_nodes].ravel()

    def expand(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Expand periodic coefficients to the original DOF layout."""
        if values.shape != (self.size,):
            raise ValueError("Expected a flat vector in the periodic DOF layout.")
        return values[self.indices]

    def accumulate(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sum an assembled FP64 vector into its periodic test DOFs."""
        if values.shape != self.indices.shape or values.dtype != np.float64:
            raise ValueError("Expected a flat FP64 vector in the original DOF layout.")
        return np.bincount(self.indices, weights=values, minlength=self.size)


def _assemble_operators(velocity, pressure, velocity_map, pressure_map, lower, upper, jit_options):
    from scipy.sparse import csr_matrix

    domain = velocity.mesh
    test = ufl.TestFunction(velocity)
    measure = ufl.Measure("dx", domain=domain)
    weights = assemble_vector(
        form(
            ufl.inner(ufl.as_vector((1.0, 1.0, 1.0)), test) * measure,
            dtype=np.float64,
            jit_options=jit_options,
        )
    )
    weights.scatter_reverse(la.InsertMode.add)
    mass = velocity_map.accumulate(weights.array)
    volume = math.prod(np.subtract(upper, lower))
    if np.any(mass <= 0) or not np.isclose(mass.sum(), 3 * volume):
        raise ValueError("Invalid periodic lumped velocity mass.")

    # The native CSR assembly interface accepts equal trial and test block
    # sizes. Assemble scalar derivative blocks with the same Q2 basis.
    basis = basix.finite_element.FiniteElement(velocity.element.basix_element)
    scalar = functionspace(domain, basix.ufl.wrap_element(basis))
    scalar_map = PeriodicDofMap(scalar, velocity_map.shape, lower, upper)
    trial, test = ufl.TrialFunction(scalar), ufl.TestFunction(pressure)
    rows, columns, values = [], [], []
    for axis in range(3):
        matrix = assemble_matrix(
            form(trial.dx(axis) * test * measure, dtype=np.float64, jit_options=jit_options)
        )
        matrix.scatter_reverse()
        block = matrix.to_scipy().tocoo()
        rows.append(pressure_map.node_indices[block.row])
        columns.append(3 * scalar_map.node_indices[block.col] + axis)
        values.append(block.data)
    divergence = csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(pressure_map.size, velocity_map.size),
    )
    divergence.eliminate_zeros()
    return mass, divergence


def _inverse_symbol(
    divergence: csr_matrix,
    gradient: csr_matrix,
    inverse_mass: npt.NDArray[np.float64],
    shape: tuple[int, int, int],
) -> npt.NDArray[np.float64]:
    impulse = np.zeros(math.prod(shape), dtype=np.float64)
    impulse[0] = 1
    column = divergence @ (inverse_mass * (gradient @ impulse))
    symbol = np.fft.rfftn(column.reshape(shape))
    scale = np.max(np.abs(symbol))
    if np.max(np.abs(symbol.imag)) > 1e-12 * scale or abs(symbol.flat[0]) > 1e-12 * scale:
        raise ValueError("Expected a symmetric pressure operator with a constant nullspace.")
    spectrum = symbol.real.copy()
    spectrum.flat[0] = 1
    if np.min(spectrum) <= 1e-13 * scale:
        raise ValueError("The pressure operator has an unexpected null or negative mode.")

    # Check that the assembled pressure operator is periodic away from the impulse.
    sample = np.random.default_rng(71).standard_normal(impulse.size)
    direct = divergence @ (inverse_mass * (gradient @ sample))
    transformed = np.fft.irfftn(
        symbol * np.fft.rfftn(sample.reshape(shape)), s=shape, axes=(0, 1, 2)
    )
    np.testing.assert_allclose(transformed.ravel(), direct, rtol=1e-10, atol=1e-11 * scale)
    inverse = 1 / spectrum
    inverse.flat[0] = 0
    return inverse


class PeriodicIncompressibleProblem:
    """Advance a periodic Q2/Q1 flow with lumped mass and a pressure projection.

    The user supplies a momentum load as a UFL form compiled by FFCx. The
    solver applies its negative mass-weighted residual and projects each
    classical RK4 stage using the assembled divergence and an FFT inverse
    of the pressure Schur complement. The method requires FP64, one MPI rank,
    a uniform hexahedral box and a single state coefficient in the load.
    The CPU backend uses the supplied Function as coefficient storage.
    """

    def __init__(
        self,
        momentum: Form,
        state: Function,
        pressure_space: FunctionSpace,
        *,
        shape: tuple[int, int, int],
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        backend: Literal["cpu", "cuda"] = "cpu",
        jit_options: dict | None = None,
    ) -> None:
        """Assemble the mass and divergence and prepare repeated RHS evaluations.

        Args:
            momentum: Compiled linear form for the velocity residual.
            state: Function representing the velocity coefficient in the form.
            pressure_space: Scalar Q1 space on the velocity mesh.
            shape: Number of hexahedral cells along each axis.
            lower: Coordinates of the lower periodic corner.
            upper: Coordinates of the upper periodic corner.
            backend: DOLFINx CPU assembly or FFCx-generated CUDA assembly on
                the current CuPy device.
            jit_options: DOLFINx JIT options, also used for the momentum form.
        """
        if backend not in ("cpu", "cuda"):
            raise ValueError(f"Unknown periodic flow backend: {backend}.")
        velocity_space = state.function_space
        if (
            momentum.rank != 1
            or momentum.function_spaces[0] != velocity_space
            or momentum.dtype != np.dtype(np.float64)
            or list(momentum._cpp_object.coefficients) != [state._cpp_object]
        ):
            raise ValueError("Expected an FP64 velocity load with the velocity as its coefficient.")
        if (
            velocity_space.mesh is not pressure_space.mesh
            or velocity_space.mesh.topology.cell_type != mesh.CellType.hexahedron
        ):
            raise ValueError("Velocity and pressure spaces must share a single hexahedral mesh.")
        if (
            velocity_space.dofmap.index_map_bs != 3
            or pressure_space.dofmap.index_map_bs != 1
            or len(shape) != 3
            or any(not isinstance(n, (int, np.integer)) or n < 2 for n in shape)
        ):
            raise ValueError("Expected three velocity components and at least two cells per axis.")
        if velocity_space.mesh.topology.index_map(3).size_global != math.prod(shape):
            raise ValueError("The mesh cell count must match the Cartesian box.")
        for space, degree in ((velocity_space, 2), (pressure_space, 1)):
            element = space.element.basix_element
            if element.family != basix.ElementFamily.P or element.degree != degree:
                raise ValueError("Expected Lagrange Q2 velocity and Q1 pressure.")

        self.velocity_dofs = PeriodicDofMap(
            velocity_space, tuple(2 * n for n in shape), lower, upper
        )
        self.pressure_dofs = PeriodicDofMap(pressure_space, shape, lower, upper)
        self._state = state
        self._momentum = momentum
        self.backend = backend
        mass, divergence = _assemble_operators(
            velocity_space,
            pressure_space,
            self.velocity_dofs,
            self.pressure_dofs,
            lower,
            upper,
            jit_options,
        )
        gradient = divergence.T.tocsr()
        inverse_mass = 1 / mass
        symbol = _inverse_symbol(divergence, gradient, inverse_mass, self.pressure_dofs.shape)
        self._cuda_assembler = None
        if backend == "cuda":
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix

            from dolfinx.fem.cuda import VectorAssembler

            self._xp = cp
            self._cuda_assembler = VectorAssembler(
                momentum,
                (state,),
                coefficient_dofs=(self.velocity_dofs.indices,),
                test_dofs=self.velocity_dofs.indices,
                jit_options=jit_options,
            )
            self.divergence = csr_matrix(divergence)
            self._gradient = csr_matrix(gradient)
        else:
            self._xp = np
            self.divergence = divergence
            self._gradient = gradient
        self.mass = self._xp.asarray(mass)
        self._inverse_mass = self._xp.asarray(inverse_mass)
        self._inverse_symbol = self._xp.asarray(symbol)

    def residual(
        self, velocity: npt.NDArray[np.float64] | cp.ndarray
    ) -> npt.NDArray[np.float64] | cp.ndarray:
        """Assemble the momentum load in the periodic velocity layout."""
        if self._cuda_assembler is not None:
            return self._cuda_assembler.assemble((velocity,))
        self._state.x.array[:] = self.velocity_dofs.expand(velocity)
        assembled = assemble_vector(self._momentum)
        assembled.scatter_reverse(la.InsertMode.add)
        return self.velocity_dofs.accumulate(assembled.array)

    def _correction(self, velocity):
        shape = self.pressure_dofs.shape
        divergence = (self.divergence @ velocity).reshape(shape)
        potential = self._xp.fft.irfftn(
            self._xp.fft.rfftn(divergence) * self._inverse_symbol, s=shape, axes=(0, 1, 2)
        ).ravel()
        return self._inverse_mass * (self._gradient @ potential), potential

    def project(
        self, velocity: npt.NDArray[np.float64] | cp.ndarray
    ) -> npt.NDArray[np.float64] | cp.ndarray:
        """Project a velocity vector onto the discrete divergence-free space."""
        correction, _ = self._correction(velocity)
        return velocity - correction

    def pressure(
        self, velocity: npt.NDArray[np.float64] | cp.ndarray
    ) -> npt.NDArray[np.float64] | cp.ndarray:
        """Recover the zero-mean pressure for the current velocity."""
        _, potential = self._correction(-self._inverse_mass * self.residual(velocity))
        return -potential

    def _derivative(self, velocity):
        return self.project(-self._inverse_mass * self.residual(velocity))

    def step(
        self, velocity: npt.NDArray[np.float64] | cp.ndarray, dt: float
    ) -> npt.NDArray[np.float64] | cp.ndarray:
        """Advance one full RK4 step with pressure projection at each stage."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("The time step must be finite and positive.")
        k1 = self._derivative(velocity)
        k2 = self._derivative(velocity + 0.5 * dt * k1)
        k3 = self._derivative(velocity + 0.5 * dt * k2)
        k4 = self._derivative(velocity + dt * k3)
        return velocity + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
