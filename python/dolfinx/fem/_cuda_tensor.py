# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate staged CUDA contractions for FFCx linear cell integrals."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

import basix
import ffcx
import ufl
from ffcx.analysis import analyze_ufl_objects
from ffcx.codegeneration import lnodes as L
from ffcx.codegeneration.backend import FFCXBackend
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.common import integral_data
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.ir.elementtables import get_modified_terminal_element
from ffcx.ir.representation import compute_ir

if TYPE_CHECKING:
    from ffcx.ir.analysis.modified_terminals import ModifiedTerminal
    from ffcx.ir.elementtables import UniqueTableReferenceT
    from ffcx.ir.representation import IntegralIR
    from ffcx.ir.representationutils import QuadratureRule

_CELLS = (basix.CellType.quadrilateral, basix.CellType.hexahedron)
_BARRIER = "    __syncthreads();\n"


class _Unsupported(Exception):
    """Integral whose original UFCx kernel should be used."""


class TensorKernel(NamedTuple):
    """A generated CUDA kernel using one block per cell."""

    source: str
    threads: int


@dataclass(frozen=True)
class _Basis:
    shape: tuple[int, ...]
    dofs: tuple[int, ...]
    tables: tuple[str, ...]


def _product_order(points: npt.NDArray[np.float64]):
    """Map the product grid back to the supplied point order."""
    axes = tuple(np.unique(axis) for axis in points.T)
    shape = tuple(map(len, axes))
    if math.prod(shape) != len(points):
        raise _Unsupported("Expected a complete product grid.")
    order = np.lexsort(tuple(points.T[::-1]))
    if not np.array_equal(points[order], np.array(list(product(*axes)))):
        raise _Unsupported("Expected a complete product grid.")
    return axes, shape, order


def affine_jacobians(
    expression: ufl.Form, coordinates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    """Cache constant Jacobians when all pullbacks are affine Q1 maps."""
    domain = expression.ufl_domain()
    element = domain.ufl_coordinate_element().basix_element
    if (
        element.cell_type not in _CELLS
        or element.family != basix.ElementFamily.P
        or element.degree != 1
        or not element.interpolation_is_identity
        or element.dim != coordinates.shape[1]
    ):
        return None

    # Sort the reference points to account for reordered DOFs.
    try:
        axes, shape, order = _product_order(element.points)
    except _Unsupported:
        return None
    if any(not np.array_equal(axis, [0, 1]) for axis in axes):
        return None
    corners = coordinates[:, order].reshape((len(coordinates), *shape, 3))
    columns = []
    for axis in range(len(shape)):
        right = corners.take(1, axis=axis + 1).reshape(len(corners), -1, 3)
        left = corners.take(0, axis=axis + 1).reshape(right.shape)
        edges = right - left
        # TwoDiff supplies the rounding error of each edge difference.
        # The high and low parts must both agree for affine maps.
        tail = edges - right
        errors = (right - (edges - tail)) - (left + tail)
        if not (np.all(edges == edges[:, :1]) and np.all(errors == errors[:, :1])):
            return None
        columns.append(edges[:, 0])
    return np.stack(columns, axis=-1)


class _Tables:
    """Check interval factors against the full FFCx tables."""

    def __init__(self, rule: QuadratureRule):
        self.axes, self.shape, order = _product_order(rule.points)
        self.weights = rule.weights[order]
        self._order = order
        self._cache: dict[tuple, _Basis] = {}
        self._factors: dict[tuple, tuple[str, npt.NDArray[np.float64]]] = {}

    def basis(self, mt: ModifiedTerminal, table: UniqueTableReferenceT) -> _Basis:
        if table.is_permuted or table.values.shape[:2] != (1, 1):
            raise _Unsupported("Expected a cell table without permutations.")
        terminal = get_modified_terminal_element(mt)
        if terminal is None or terminal.averaged:
            raise _Unsupported("Expected a basis or basis derivative.")
        component, _, _ = terminal.element.get_component_element(terminal.fc)
        if isinstance(component, basix.ufl._ComponentElement):
            component = component.basix_element
        if not isinstance(component, basix.ufl._BasixElement):
            raise _Unsupported("Expected a scalar Basix component.")
        element = component.basix_element
        if (
            element.cell_type not in _CELLS
            or element.family != basix.ElementFamily.P
            or element.value_shape
            or element.map_type != basix.MapType.identity
            or not element.interpolation_is_identity
            or not element.dof_transformations_are_permutations
        ):
            raise _Unsupported("Expected an identity-mapped nodal tensor element.")

        derivatives = terminal.local_derivatives
        key = (element, derivatives, table.name)
        if key in self._cache:
            return self._cache[key]
        axes, shape, dofs = _product_order(element.points)
        if any(n != element.degree + 1 for n in shape):
            raise _Unsupported("Expected a complete nodal tensor basis.")
        interval = basix.create_element(
            element.family,
            basix.CellType.interval,
            element.degree,
            element.lagrange_variant,
            element.dpc_variant,
            element.discontinuous,
            dtype=element.dtype,
        )
        interval_order = np.argsort(interval.points[:, 0])
        nodes = interval.points[interval_order, 0]
        if any(not np.allclose(axis, nodes, rtol=0, atol=2e-14) for axis in axes):
            raise _Unsupported("Interval interpolation points differ from the original element.")

        factors = [
            interval.tabulate(d, axis[:, None])[d, :, :, 0][:, interval_order]
            for axis, d in zip(self.axes, derivatives, strict=True)
        ]
        full = factors[0]
        for factor in factors[1:]:
            full = np.einsum("qi,rj->qrij", full, factor).reshape(
                full.shape[0] * factor.shape[0], full.shape[1] * factor.shape[1]
            )
        if len(dofs) != table.values.shape[-1]:
            raise _Unsupported("Unexpected number of DOFs in the FFCx table.")
        expected = table.values[0, 0][:, dofs]
        if expected.shape[0] == 1:
            expected = np.broadcast_to(expected, full.shape)
        else:
            expected = expected[self._order]
        if full.shape != expected.shape or not np.allclose(full, expected, atol=5e-14, rtol=5e-14):
            raise _Unsupported("Interval factors differ from the original FFCx table.")

        names = []
        for values in factors:
            key_values = (values.shape, values.tobytes())
            if key_values not in self._factors:
                self._factors[key_values] = (f"basis_{len(self._factors)}", values)
            names.append(self._factors[key_values][0])
        result = _Basis(shape, tuple(dofs.tolist()), tuple(names))
        self._cache[key] = result
        return result

    def declarations(self, formatter: Formatter) -> str:
        arrays = [*self._factors.values(), ("weights", self.weights)]
        return "".join(
            "__device__ "
            + formatter(L.ArrayDecl(L.Symbol(name, dtype=L.DataType.REAL), values=data, const=True))
            for name, data in arrays
        )


@dataclass
class _Buffer:
    size: int
    first: int
    last: int


class _Schedule:
    """Schedule gathers, contractions and scatters around barriers."""

    def __init__(self, quadrature: tuple[int, ...]):
        self.quadrature = quadrature
        self.initialization: list[str] = []
        self.forward: dict[int, list[str]] = defaultdict(list)
        self.transpose: dict[int, list[str]] = defaultdict(list)
        self.device_tables: list[str] = []
        self._buffers: dict[str, _Buffer] = {}
        self._gathers: dict[tuple, str] = {}
        self._prefixes: dict[tuple, str] = {}
        self.pointwise_phase = len(quadrature) + 1

    def buffer(self, size: int, phase: int) -> str:
        name = f"buf_{len(self._buffers)}"
        self._buffers[name] = _Buffer(size, phase, phase)
        return name

    def read(self, name: str, phase: int) -> None:
        buffer = self._buffers[name]
        buffer.last = max(buffer.last, phase)

    def indices(self, indices: tuple[int, ...]) -> str:
        name = f"map_{len(self.device_tables)}"
        values = ", ".join(map(str, indices))
        self.device_tables.append(f"__device__ const int {name}[{len(indices)}] = {{{values}}};\n")
        return name

    def gather(self, source: str, dofs: tuple[int, ...]) -> str:
        key = (source, dofs)
        if key not in self._gathers:
            slot = self.buffer(len(dofs), 0)
            indices = self.indices(dofs)
            load = source.format(dof=f"{indices}[i]")
            self.initialization.append(
                f"""    for (int i = threadIdx.x; i < {len(dofs)}; i += blockDim.x)
        {slot}[i] = {load};
"""
            )
            self._gathers[key] = slot
        return self._gathers[key]

    @staticmethod
    def _contract(
        shape: tuple[int, ...],
        axis: int,
        target: int,
        operands: list[tuple[str, str]],
        store: str,
        *,
        transpose: bool,
    ) -> str:
        width = shape[axis]
        stride = math.prod(shape[axis + 1 :])
        size = math.prod(shape) // width * target
        access = "[k][j]" if transpose else "[j][k]"
        accumulate = "\n".join(
            f"            value += {source}[(prefix * {width} + k) * {stride} + suffix]"
            f" * {table}{access};"
            for source, table in operands
        )
        return f"""    for (int i = threadIdx.x; i < {size}; i += blockDim.x)
    {{
        const int j = i / {stride} % {target};
        const int prefix = i / {stride * target};
        const int suffix = i % {stride};
        double value = 0.0;
        #pragma unroll
        for (int k = 0; k < {width}; ++k)
        {{
{accumulate}
        }}
        {store}
    }}
"""

    def interpolate(self, source: str, basis: _Basis) -> str:
        shape = basis.shape
        for axis, (table, target) in enumerate(zip(basis.tables, self.quadrature, strict=True)):
            key = (axis, source, table)
            next_shape = (*shape[:axis], target, *shape[axis + 1 :])
            if key not in self._prefixes:
                self.read(source, axis + 1)
                output = self.buffer(math.prod(next_shape), axis + 1)
                self.forward[axis].append(
                    self._contract(
                        shape,
                        axis,
                        target,
                        [(source, table)],
                        f"{output}[i] = value;",
                        transpose=False,
                    )
                )
                self._prefixes[key] = output
            source, shape = self._prefixes[key], next_shape
        self.read(source, self.pointwise_phase)
        return source

    def scatter(
        self, dofs: tuple[int, ...], test_shape: tuple[int, ...], terms: list[tuple]
    ) -> None:
        """Apply the transposed test basis with shared contractions."""
        shape = self.quadrature
        indices = self.indices(dofs)
        remaining = terms
        for axis in reversed(range(len(shape))):
            groups = defaultdict(list)
            for source, tables in remaining:
                groups[tables[:-1]].append((source, tables[-1]))
            target = test_shape[axis]
            next_shape = (*shape[:axis], target, *shape[axis + 1 :])
            remaining = []
            phase = self.pointwise_phase + len(shape) - axis
            for prefix, operands in groups.items():
                for source, _ in operands:
                    self.read(source, phase)
                if axis:
                    output = self.buffer(math.prod(next_shape), phase)
                    remaining.append((output, prefix))
                    store = f"{output}[i] = value;"
                else:
                    store = f"atomicAdd(output + dofs[cell * output_width + {indices}[i]], value);"
                self.transpose[axis].append(
                    self._contract(shape, axis, target, operands, store, transpose=True)
                )
            shape = next_shape

    def shared(self, limit: int) -> str:
        """Reuse space only after the last reading phase has ended."""
        placed: list[tuple[int, _Buffer]] = []
        declarations = []
        words = 0
        for name, buffer in sorted(self._buffers.items(), key=lambda item: -item[1].size):
            start = 0
            for offset, other in sorted(placed, key=lambda item: item[0]):
                if other.last < buffer.first or buffer.last < other.first:
                    continue
                if start + buffer.size <= offset:
                    break
                start = max(start, offset + other.size)
            placed.append((start, buffer))
            words = max(words, start + buffer.size)
            declarations.append(f"    double* const {name} = scratch + {start};\n")
        if 8 * words > limit:
            raise _Unsupported("Shared staging buffers exceed the device limit.")
        return f"    __shared__ double scratch[{words}];\n" + "".join(declarations)

    @property
    def interpolation(self) -> str:
        blocks = [self.initialization, *[self.forward[a] for a in sorted(self.forward)]]
        return "".join("".join(block) + _BARRIER for block in blocks if block)

    @property
    def integration(self) -> str:
        return _BARRIER.join(
            "".join(self.transpose[a]) for a in sorted(self.transpose, reverse=True)
        )


def _emit(
    ir: IntegralIR,
    options: dict,
    coefficient_widths: tuple[int, ...],
    output_width: int,
    coordinate_width: int,
    affine_width: int | None,
    shared_limit: int,
) -> tuple[TensorKernel, bool]:
    if ir.rank != 1 or len(ir.expression.integrand) != 1:
        raise _Unsupported("Expected one quadrature rule for a linear form.")
    (domain, rule), integrand = next(iter(ir.expression.integrand.items()))
    if (
        domain not in _CELLS
        or rule.has_tensor_factors
        or ir.expression.needs_facet_permutations
        or tuple(ir.expression.tensor_shape) != (output_width,)
    ):
        raise _Unsupported("Unsupported cell integral or test space.")
    graph = integrand["factorization"]
    terminals = [
        (node, mt)
        for node in graph.nodes.values()
        if node["status"] != "inactive"
        and not node["expression"]._ufl_is_literal_
        and (mt := node.get("mt")) is not None
    ]
    if any(
        not isinstance(
            mt.terminal, (ufl.Coefficient, ufl.Constant, ufl.Jacobian, ufl.SpatialCoordinate)
        )
        for _, mt in terminals
    ):
        raise _Unsupported("The integrand needs additional UFCx terminal definitions.")

    tables = _Tables(rule)
    schedule = _Schedule(tables.shape)
    dimension = len(tables.shape)
    points = len(rule.weights)
    # The assembler fixes the geometry during preparation.
    use_jacobians = affine_width == 3 * dimension and all(
        isinstance(mt.terminal, (ufl.Coefficient, ufl.Constant))
        or (isinstance(mt.terminal, ufl.Jacobian) and not mt.local_derivatives)
        for _, mt in terminals
    )
    backend = FFCXBackend(ir, options)
    generator = IntegralGenerator(ir, backend)
    formatter = Formatter(np.float64)
    defined: dict[str, str] = {}
    for node, mt in terminals:
        terminal, table = mt.terminal, node.get("tr")
        if isinstance(terminal, ufl.Constant):
            continue
        if table is None or table.offset is None or table.block_size is None:
            raise _Unsupported("Missing coefficient or geometry table.")
        access = backend.access.get(mt, table, rule)
        offset = table.offset
        if isinstance(terminal, ufl.Coefficient):
            index = ir.expression.coefficient_numbering[terminal]
            width = coefficient_widths[index]
            source = f"coefficient{index}[map{index}[cell * {width} + {{dof}}]]"
        else:
            width = coordinate_width
            source = "geometry[{dof}]"
        if isinstance(access, L.Symbol):
            basis = tables.basis(mt, table)
            if use_jacobians and isinstance(terminal, ufl.Jacobian):
                component, axis = mt.component
                read = f"geometry[{component * dimension + axis}]"
            else:
                stride = table.block_size if isinstance(terminal, ufl.Coefficient) else 3
                dofs = tuple(offset + stride * i for i in basis.dofs)
                if not all(0 <= i < width for i in dofs):
                    raise _Unsupported("Coefficient or geometry DOFs exceed the input layout.")
                value = schedule.interpolate(schedule.gather(source, dofs), basis)
                read = f"{value}[iq]"
        elif isinstance(terminal, ufl.Coefficient) and table.ttype == "ones":
            # FFCx uses a direct DOF read for piecewise constants.
            if not 0 <= offset < width or table.values.shape[-1] != 1:
                raise _Unsupported("Unsupported direct coefficient access.")
            name = f"coefficient_{index}_dof_{offset}"
            access = L.Symbol(name, dtype=L.DataType.SCALAR)
            read = source.format(dof=offset)
        else:
            raise _Unsupported("Unsupported terminal access.")
        if access.name in defined and defined[access.name] != read:
            raise _Unsupported("Generated terminal symbols overlap.")
        defined[access.name] = read
        scope = (rule, domain) if node["status"] == "varying" else (None, None)
        generator.set_var(*scope, node["expression"], access)

    piece_declarations, pieces = generator.generate_piecewise_partition(rule, domain)
    point_declarations, pointwise = generator.generate_varying_partition(rule, domain)
    if piece_declarations or point_declarations:
        raise _Unsupported("The integrand requires additional definitions.")

    groups: dict[tuple, list] = defaultdict(list)
    factor_values: dict[int, str] = {}
    for blockmap, blocks in integrand["block_contributions"].items():
        for block in blocks:
            if len(blockmap) != 1 or len(block.factor_indices_comp_indices) != 1:
                raise _Unsupported("Expected a rank-one contraction.")
            (argument,) = block.ma_data
            mt = integrand["modified_arguments"][argument.ma_index]
            basis = tables.basis(mt, argument.tabledata)
            dofs = tuple(blockmap[0][i] for i in basis.dofs)
            if not all(0 <= i < output_width for i in dofs):
                raise _Unsupported("Test DOFs exceed the output layout.")
            index = block.factor_indices_comp_indices[0][0]
            if index not in factor_values:
                factor_values[index] = schedule.buffer(points, schedule.pointwise_phase)
            groups[basis.shape, dofs].append((factor_values[index], basis.tables))

    if not groups:
        raise _Unsupported("No test basis contractions to schedule.")
    for (shape, dofs), terms in groups.items():
        schedule.scatter(dofs, shape, terms)
    shared = schedule.shared(shared_limit)
    terminal_reads = "".join(
        f"        const double {name} = {read};\n" for name, read in defined.items()
    )

    factors = []
    for index, output in factor_values.items():
        expression = generator.get_var(rule, domain, graph.nodes[index]["expression"])
        factors.append(f"        {output}[iq] = weights[iq] * ({formatter(expression)});\n")
    arguments = "".join(
        f", const int* map{i}, const double* coefficient{i}" for i in range(len(coefficient_widths))
    )
    geometry_width = affine_width if use_jacobians else coordinate_width
    code = f"""{tables.declarations(formatter)}{"".join(schedule.device_tables)}
extern "C" __global__ void assemble_vector(
    int count, const int* dofs, const double* coordinates, const double* c,
    double* output{arguments})
{{
    const int cell = blockIdx.x;
    constexpr int output_width = {output_width};
    const double* geometry = coordinates + cell * {geometry_width};
{shared}{schedule.interpolation}
    for (int iq = threadIdx.x; iq < {points}; iq += blockDim.x)
    {{
{terminal_reads}{formatter(L.StatementList(pieces + pointwise))}{"".join(factors)}
    }}
{_BARRIER}{schedule.integration}}}
"""
    threads = max(32, min(128, ((max(points, output_width) + 31) // 32) * 32))
    return TensorKernel(code, threads), use_jacobians


def kernels(
    expression: ufl.Form,
    compiler_options: dict,
    integral_ids: tuple[int, ...],
    coefficient_positions: tuple[int, ...],
    coefficient_widths: tuple[int, ...],
    output_width: int,
    coordinate_width: int,
    affine_width: int | None,
    shared_limit: int,
) -> dict[int, tuple[TensorKernel, bool]]:
    """Generate supported kernels in the original UFCx cell order."""
    if len(expression.arguments()) != 1 or not integral_ids:
        return {}
    options = ffcx.get_options(compiler_options)
    analysis = analyze_ufl_objects(
        [expression],
        options["scalar_type"],
        do_cancel_jacobian_products=options["do_cancel_jacobian_products"],
    )
    ir = compute_ir(analysis, {}, "", options, None)
    form = ir.forms[0]
    metadata = integral_data(form)
    first, last = metadata.offsets[:2]
    if (
        tuple(metadata.ids[first:last]) != integral_ids
        or tuple(form.original_coefficient_positions) != coefficient_positions
    ):
        return {}
    integrals = {integral.expression.name: integral for integral in ir.integrals}
    sources = {}
    for index, name in enumerate(metadata.names[first:last]):
        try:
            sources[index] = _emit(
                integrals[name],
                options,
                coefficient_widths,
                output_width,
                coordinate_width,
                affine_width,
                shared_limit,
            )
        except _Unsupported:
            continue
    return sources
