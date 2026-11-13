# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CUDA vector assembly for generated cell integrals."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from dolfinx import jit
from dolfinx.fem import Form, Function, FunctionSpace, IntegralType, pack_constants
from dolfinx.mesh import CellType

if TYPE_CHECKING:
    import cupy as cp

__all__ = ["VectorAssembler"]


class _Launch(NamedTuple):
    kernel: cp.RawKernel
    blocks: int
    threads: int
    count: int
    dofs: cp.ndarray
    geometry: cp.ndarray
    maps: tuple[cp.ndarray, ...]


def _cell_kernels(source: str) -> list[tuple[str, str]]:
    """Read generated FP64 kernel definitions in UFCx integral order."""
    integrals = re.search(
        r"static ufcx_integral\s*\*\s*form_integrals_\w+\[\d+\]\s*=\s*\{([^{}]+)\};",
        source,
    )
    if integrals is None:
        raise ValueError("Could not read integral factories from the generated FFCx form.")
    factories = dict(re.findall(r"ufcx_integral\s+(\w+)\s*=\s*\{(.*?)\};", source, re.DOTALL))
    kernels = []
    for factory in re.findall(r"&\s*(\w+)", integrals[1]):
        entry = re.search(r"\.tabulate_tensor_float64\s*=\s*(\w+)", factories[factory])
        if entry is None:
            raise ValueError(f"Could not find the FP64 kernel for {factory}.")
        name = entry[1]
        match = re.search(rf"\bvoid\s+{name}\s*\([^{{}};]*\)\s*\{{", source)
        if match is None:
            raise ValueError(f"Could not find the FFCx definition of {name}.")
        start, end = match.start(), match.end()
        depth = 1
        while depth and end < len(source):
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        if depth:
            raise ValueError("Incomplete FFCx kernel definition.")
        function = re.sub(r"\brestrict\b", "__restrict__", source[start:end])
        function = function.replace("_Alignas(", "alignas(")
        kernels.append((name, "__device__ " + function))
    return kernels


def _cell_wrapper(
    name: str, widths: tuple[int, ...], output_width: int, coordinate_width: int
) -> str:
    """Wrap the UFCx cell kernel with device gathers and scatters."""
    arguments = []
    gather = []
    offset = 0
    for i, width in enumerate(widths):
        arguments.extend((f"const int* map{i}", f"const double* coefficient{i}"))
        gather.append(
            f"for (int j = 0; j < {width}; ++j)\n"
            f"        w[{offset} + j] = coefficient{i}[map{i}[cell * {width} + j]];"
        )
        offset += width
    coefficient_args = "".join(f", {argument}" for argument in arguments)
    coefficient_loads = "\n    ".join(gather)
    return f"""
extern "C" __global__ void assemble_vector(
    int count, const int* dofs, const double* coordinates,
    const double* constants, double* output{coefficient_args})
{{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= count)
        return;
    double w[{max(offset, 1)}];
    double local[{output_width}] = {{}};
    {coefficient_loads}
    {name}(local, w, constants, coordinates + cell * {coordinate_width},
        nullptr, nullptr, nullptr);
    for (int j = 0; j < {output_width}; ++j)
        atomicAdd(output + dofs[cell * {output_width} + j], local[j]);
}}
"""


class VectorAssembler:
    """Assemble an FFCx linear form's vector on one CUDA device.

    The form uses real FP64, cell integrals and one mesh on a serial MPI
    communicator. Test and coefficient elements must have dofmaps that
    encode their local ordering without additional DOF transformations.
    Every generated integral must have a cell integration domain.
    Mesh coordinates, domains and index maps are fixed at construction.
    Pass current coefficient values in device arrays.

    Supported nodal quadrilateral and hexahedral forms use staged tensor
    contractions generated from FFCx's quadrature tables and integrand.
    Other forms use the original UFCx cell kernel on CUDA.

    Optional DOF maps associate each unrolled DOLFINx DOF with an index
    in the device vector. A many-to-one map assembles directly into a
    constrained space, for example when identifying periodic DOFs.
    """

    def __init__(
        self,
        form: Form,
        coefficients: tuple[Function, ...],
        *,
        coefficient_dofs: tuple[npt.NDArray[np.integer] | None, ...] | None = None,
        test_dofs: npt.NDArray[np.integer] | None = None,
        jit_options: dict | None = None,
    ) -> None:
        """Prepare index maps and compile CUDA kernels for a fixed form.

        Args:
            form: Compiled rank-one FFCx form.
            coefficients: DOLFINx functions in the form's UFCx coefficient
                order. Supply device arrays in the same order.
            coefficient_dofs: Input index maps in UFCx coefficient order.
            test_dofs: Optional output index map for the test space.
            jit_options: DOLFINx JIT options used to create the form.
        """
        import cupy as cp

        if cp.cuda.runtime.is_hip:
            raise RuntimeError("CUDA vector assembly requires an NVIDIA CUDA build of CuPy.")
        if form.rank != 1 or form.dtype != np.dtype(np.float64):
            raise ValueError("CUDA vector assembly requires a real FP64 linear form.")
        if form.mesh.comm.size != 1 or len(form.mesh.geometry.dofmaps) != 1:
            raise ValueError("CUDA vector assembly requires a serial mesh with one cell type.")
        if form.mesh.geometry.x.dtype != np.float64:
            raise ValueError("CUDA vector assembly requires FP64 geometry.")
        if form.integral_types != {IntegralType.cell} or form.ufcx_form is None:
            raise ValueError("CUDA vector assembly requires generated cell integrals.")
        cpp_coefficients = form._cpp_object.coefficients
        if len(coefficients) != len(cpp_coefficients) or any(
            function._cpp_object is not expected
            for function, expected in zip(coefficients, cpp_coefficients, strict=True)
        ):
            raise ValueError("Pass the form's coefficient functions in UFCx order.")

        self._cp = cp
        self._form = form
        self._device = cp.cuda.Device()
        self._num_constants = pack_constants(form).size
        maps = (None,) * len(coefficients) if coefficient_dofs is None else coefficient_dofs
        self._coefficient_sizes = []
        coefficient_indices = []
        for coefficient, mapping in zip(coefficients, maps, strict=True):
            indices, size = self._cell_dofs(coefficient.function_space, mapping)
            coefficient_indices.append(indices)
            self._coefficient_sizes.append(size)
        self._coefficient_sizes = tuple(self._coefficient_sizes)
        indices, self.size = self._cell_dofs(form.function_spaces[0], test_dofs)
        self._kernels = []
        geometry = form.mesh.geometry
        coordinates = np.ascontiguousarray(geometry.x[geometry.dofmaps[0]])
        coordinate_width = coordinates.shape[1] * coordinates.shape[2]
        output_width = indices.shape[1]

        cell_count = form.ufcx_form.form_integral_offsets[1]
        if cell_count != form.num_integrals(IntegralType.cell, 0):
            raise ValueError("Expected every generated integral to have a cell integration domain.")

        widths = tuple(mapping.shape[1] for mapping in coefficient_indices)
        tensor_kernels = {}
        jacobians = None
        if form._ufl_data is not None and form.mesh.topology.cell_type in (
            CellType.quadrilateral,
            CellType.hexahedron,
        ):
            from dolfinx.fem import _cuda_tensor

            expression, options = form._ufl_data
            jacobians = _cuda_tensor.affine_jacobians(expression, coordinates)
            tensor_kernels = _cuda_tensor.kernels(
                expression,
                options,
                tuple(form.ufcx_form.form_integral_ids[i] for i in range(cell_count)),
                tuple(form.ufcx_form.original_coefficient_positions[i] for i in range(len(widths))),
                widths,
                output_width,
                coordinate_width,
                None if jacobians is None else jacobians.shape[1] * jacobians.shape[2],
                min(
                    48 * 1024,
                    cp.cuda.runtime.getDeviceProperties(self.device_id)["sharedMemPerBlock"],
                ),
            )

        original_kernels = None
        for index in range(cell_count):
            cells = form._cpp_object.domains(IntegralType.cell, index)
            count = len(cells)
            if not count:
                continue
            kernel, data = None, coordinates
            if candidate := tensor_kernels.get(index):
                tensor, use_jacobians = candidate
                try:
                    kernel = cp.RawModule(
                        code=tensor.source, options=("--std=c++17", "--fmad=true")
                    ).get_function("assemble_vector")
                except cp.cuda.compiler.CompileException:
                    logging.getLogger(__name__).debug(
                        "The CUDA tensor kernel did not compile.", exc_info=True
                    )
                else:
                    blocks, threads = count, tensor.threads
                    if use_jacobians:
                        data = jacobians
            if kernel is None:
                if original_kernels is None:
                    code = form.code
                    source = (
                        code
                        if isinstance(code, str)
                        else "\n".join(part for part in code or () if part)
                    )
                    if not source:
                        cache = Path(jit.get_options(jit_options)["cache_dir"])
                        source = (cache / (form.module.__name__ + ".c")).read_text(encoding="utf-8")
                    original_kernels = _cell_kernels(source)
                    if len(original_kernels) != cell_count:
                        raise ValueError("Expected one UFCx kernel per cell integration domain.")
                name, function = original_kernels[index]
                wrapper = _cell_wrapper(name, widths, output_width, coordinate_width)
                kernel = cp.RawModule(
                    code="using uint8_t = unsigned char;\n" + function + wrapper,
                    options=("--std=c++17", "--fmad=false"),
                ).get_function("assemble_vector")
                threads = 128
                blocks = (count + threads - 1) // threads
            self._kernels.append(
                _Launch(
                    kernel,
                    blocks,
                    threads,
                    count,
                    cp.asarray(indices[cells]),
                    cp.asarray(data[cells]),
                    tuple(cp.asarray(mapping[cells]) for mapping in coefficient_indices),
                )
            )

    def _cell_dofs(
        self, space: FunctionSpace, mapping: npt.NDArray[np.integer] | None
    ) -> tuple[npt.NDArray[np.int32], int]:
        if space.mesh._cpp_object is not self._form.mesh._cpp_object:
            raise ValueError("CUDA cell integration requires spaces on the integration mesh.")
        if len(space.dofmaps) != 1 or space.element.needs_dof_transformations:
            raise ValueError("CUDA vector assembly requires a dofmap without DOF transformations.")
        dofmap = space.dofmaps[0]
        size = dofmap.index_map.size_local * dofmap.index_map_bs
        if dofmap.index_map.num_ghosts:
            raise ValueError("CUDA vector assembly requires a serial dofmap.")
        if mapping is not None:
            mapping = np.asarray(mapping)
            if (
                mapping.shape != (size,)
                or not np.issubdtype(mapping.dtype, np.integer)
                or np.any(mapping < 0)
                or np.any(mapping >= size)
            ):
                raise ValueError("Invalid CUDA vector index map.")
            size = int(mapping.max()) + 1 if mapping.size else 0
        if size > np.iinfo(np.int32).max:
            raise ValueError("CUDA vector indices must fit in int32.")
        blocks = dofmap.list
        if dofmap.bs == 1:
            indices = blocks
        else:
            indices = (
                dofmap.bs * blocks[..., None] + np.arange(dofmap.bs, dtype=np.int32)
            ).reshape(blocks.shape[0], blocks.shape[1] * dofmap.bs)
        return np.ascontiguousarray(
            indices if mapping is None else mapping[indices], dtype=np.int32
        ), size

    @property
    def device_id(self) -> int:
        """CUDA device that owns the prepared form data."""
        return self._device.id

    def _check_array(self, array: cp.ndarray, count: int, name: str) -> None:
        if (
            not isinstance(array, self._cp.ndarray)
            or array.dtype != np.float64
            or array.shape != (count,)
            or not array.flags.c_contiguous
            or array.device.id != self.device_id
        ):
            raise ValueError(f"{name} must be a contiguous FP64 vector on device {self.device_id}.")

    def assemble(
        self, coefficients: tuple[cp.ndarray, ...], *, constants: cp.ndarray | None = None
    ) -> cp.ndarray:
        """Assemble a CUDA vector from the current coefficient arrays.

        Args:
            coefficients: FP64 contiguous GPU vectors in UFCx order.
            constants: Packed FP64 device constants. By default, upload
                the current constants from the DOLFINx form.

        Returns:
            The assembled vector on the device that owns the form data.
        """
        if self._cp.cuda.Device().id != self.device_id:
            raise ValueError("Select the CUDA device that owns this assembler.")
        if len(coefficients) != len(self._coefficient_sizes):
            raise ValueError("Incorrect number of coefficient vectors.")
        for array, size in zip(coefficients, self._coefficient_sizes, strict=True):
            self._check_array(array, size, "Each coefficient")
        if constants is None:
            constants = self._cp.asarray(pack_constants(self._form))
        self._check_array(constants, self._num_constants, "Packed constants")
        output = self._cp.zeros(self.size, dtype=np.float64)
        for kernel, blocks, threads, count, dofs, geometry, maps in self._kernels:
            data = tuple(value for pair in zip(maps, coefficients, strict=True) for value in pair)
            kernel(
                (blocks,),
                (threads,),
                (np.int32(count), dofs, geometry, constants, output, *data),
            )
        return output
