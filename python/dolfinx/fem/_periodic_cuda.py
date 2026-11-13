# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-device CUDA execution for periodic Cartesian projected flow."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

import cupy as cp
import numpy as np
import numpy.typing as npt
from cupyx.scipy.sparse import csr_matrix as cuda_csr

from dolfinx.fem import Form, Function, pack_constants
from dolfinx.fem.cuda import VectorAssembler
from dolfinx.fem.periodic import _solve_pressure

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from dolfinx.fem.periodic import PeriodicDofMap


_Parts = tuple[cp.ndarray, cp.ndarray]
_PEER_ACCESS_ALREADY_ENABLED = 704


class _Layout(NamedTuple):
    first_cell: int
    cells: np.ndarray
    owned: slice
    remap: np.ndarray


def _layouts(state: Function, dofs: PeriodicDofMap, split: int) -> tuple[_Layout, _Layout]:
    """Map serial DOFs to each slab and its positive-x ghost plane."""
    n = dofs.shape[0]
    plane = math.prod(dofs.shape[1:]) * dofs.block_size
    limits = (0, 2 * split, n)
    geometry = state.function_space.mesh.geometry
    cell_centers = geometry.x[geometry.dofmaps[0], 0].mean(axis=1)
    x = geometry.x[:, 0]
    cut = x.min() + (x.max() - x.min()) * (2 * split / n)
    left = cell_centers < cut
    cells_per_plane = (dofs.shape[1] // 2) * (dofs.shape[2] // 2)
    layouts = []
    for i, mask in enumerate((left, ~left)):
        start, stop = limits[i : i + 2]
        cells = np.flatnonzero(mask)
        local_dofs = dofs.block_size * state.function_space.dofmap.list[cells, :, None]
        local_dofs = local_dofs + np.arange(dofs.block_size)
        remap = np.ascontiguousarray((dofs.indices - start * plane) % dofs.size)
        if (
            len(cells) != (stop - start) // 2 * cells_per_plane
            or remap[local_dofs].max(initial=-1) >= (stop - start + 1) * plane
        ):
            raise ValueError("The mesh cells do not match the periodic slab layout.")
        # The assembler validates the entire map, including unused DOFs.
        remap[remap >= (stop - start + 1) * plane] = 0
        layouts.append(_Layout(start // 2, cells, slice(start * plane, stop * plane), remap))
    return layouts[0], layouts[1]


class _Slab:
    def __init__(
        self,
        device: cp.cuda.Device,
        layout: _Layout,
        momentum: Form,
        state: Function,
        mass: npt.NDArray[np.float64],
        divergence: csr_matrix,
        pressure_shape: tuple[int, int, int],
        plane: int,
        jit_options: dict | None,
    ) -> None:
        self.device = device
        self.layout = layout
        self.owned = layout.owned.stop - layout.owned.start
        self.plane = plane
        # Every owned Q2 node contributes to the adjacent Q1 planes only.
        pressure_plane = math.prod(pressure_shape[1:])
        first = layout.first_cell
        stop = first + self.owned // (2 * plane)
        rows = np.arange(first * pressure_plane, stop * pressure_plane)
        ghosts = np.unique(((first - 1) % pressure_shape[0], stop % pressure_shape[0]))
        ghost_rows = (ghosts[:, None] * pressure_plane + np.arange(pressure_plane)).ravel()
        row_map = np.concatenate((rows, ghost_rows))
        columns = divergence[:, layout.owned].tocsr()
        block = columns[row_map].tocsr()
        if block.nnz != columns.nnz:
            raise ValueError("The divergence couples pressure outside adjacent slabs.")
        self.pressure_rows = slice(first * pressure_plane, stop * pressure_plane)
        self.pressure_order = row_map
        self.own_pressure = len(rows)
        self.ghost_pressure = len(ghost_rows)
        with device:
            self.stream = cp.cuda.Stream(non_blocking=True)
            with self.stream:
                self.assembler = VectorAssembler(
                    momentum,
                    (state,),
                    coefficient_dofs=(layout.remap,),
                    test_dofs=layout.remap,
                    cells=layout.cells,
                    jit_options=jit_options,
                )
                self.divergence = cuda_csr(block)
                self.gradient = cuda_csr(block.T.tocsr())
                self.mass = cp.asarray(mass[layout.owned])
                self.inverse_mass = cp.asarray(1 / mass[layout.owned])
                self.extended = cp.empty(self.owned + plane, dtype=np.float64)
                self.remote_load = cp.empty(plane, dtype=np.float64)
                self.pressure_indices = cp.asarray(row_map)
                self.pressure = cp.empty(len(row_map), dtype=np.float64)
            self.stream.synchronize()


class _Transfer:
    """Copy contiguous arrays between two compute streams."""

    def __init__(
        self,
        sender: _Slab,
        receiver: _Slab,
        count: int,
        peer: bool,
        profile: bool,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.nbytes = count * np.dtype(np.float64).itemsize
        with sender.device:
            self._host = None if peer else cp.cuda.alloc_pinned_memory(self.nbytes)
        self._pending = False
        self._profile = profile
        self._samples: list[tuple[str, cp.cuda.Event, cp.cuda.Event]] = []
        with sender.device:
            self._ready = cp.cuda.Event(disable_timing=True)
        with receiver.device:
            self._consumed = cp.cuda.Event(disable_timing=True)

    def send(self, array: cp.ndarray) -> None:
        """Queue the sending leg on the producer's stream."""
        with self.sender.device:
            stream = self.sender.stream
            if self._pending and self._host is not None:
                stream.wait_event(self._consumed)
            if self._profile and self._host is not None:
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record(stream)
            if self._host is not None:
                cp.cuda.runtime.memcpyAsync(
                    self._host.ptr,
                    array.data.ptr,
                    self.nbytes,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )
            if self._profile and self._host is not None:
                stop.record(stream)
                self._samples.append(("device_to_host", start, stop))
            self._ready.record(stream)

    def receive(self, source: cp.ndarray, destination: cp.ndarray) -> None:
        """Order the receiving leg before any dependent CUDA work."""
        with self.receiver.device:
            stream = self.receiver.stream
            if self._profile:
                wait, start, stop = cp.cuda.Event(), cp.cuda.Event(), cp.cuda.Event()
                wait.record(stream)
            stream.wait_event(self._ready)
            if self._profile:
                start.record(stream)
            runtime = cp.cuda.runtime
            if self._host is None:
                runtime.memcpyPeerAsync(
                    destination.data.ptr,
                    self.receiver.device.id,
                    source.data.ptr,
                    self.sender.device.id,
                    self.nbytes,
                    stream.ptr,
                )
            else:
                runtime.memcpyAsync(
                    destination.data.ptr,
                    self._host.ptr,
                    self.nbytes,
                    runtime.memcpyHostToDevice,
                    stream.ptr,
                )
            if self._profile:
                stop.record(stream)
                leg = "peer" if self._host is None else "host_to_device"
                self._samples.append((leg, start, stop))
                self._samples.append(("producer_wait", wait, start))
                self._samples.append(("receiver_window", wait, stop))
            self._consumed.record(stream)
        self._pending = True

    def protect_source(self) -> None:
        """Keep peer reads ahead of source buffer reuse."""
        if self._host is None:
            with self.sender.device:
                self.sender.stream.wait_event(self._consumed)

    def timings(self) -> dict[str, float]:
        """Collect this batch after both streams have finished."""
        seconds: dict[str, float] = {}
        for leg, start, stop in self._samples:
            device = self.sender.device if leg == "device_to_host" else self.receiver.device
            with device:
                elapsed = cp.cuda.get_elapsed_time(start, stop) / 1000
            seconds[leg] = seconds.get(leg, 0.0) + elapsed
        self._samples.clear()
        return seconds


def _exchange(pairs) -> None:
    # Queue every producer before making either stream wait on the other.
    for transfer, source, _ in pairs:
        transfer.send(source)
    for transfer, source, destination in pairs:
        transfer.receive(source, destination)
    for transfer, _, _ in pairs:
        transfer.protect_source()


def _peer_access(devices: tuple[cp.cuda.Device, cp.cuda.Device]) -> bool:
    runtime = cp.cuda.runtime
    directions = ((devices[0], devices[1]), (devices[1], devices[0]))
    if not all(runtime.deviceCanAccessPeer(a.id, b.id) for a, b in directions):
        return False
    for receiver, sender in directions:
        with receiver:
            try:
                runtime.deviceEnablePeerAccess(sender.id)
            except runtime.CUDARuntimeError as error:
                if error.status != _PEER_ACCESS_ALREADY_ENABLED:
                    raise
    return True


class DualCudaFlow:
    """Keep the RK4 state resident and exchange only its dependencies."""

    def __init__(
        self,
        momentum: Form,
        state: Function,
        velocity_dofs: PeriodicDofMap,
        pressure_dofs: PeriodicDofMap,
        mass: npt.NDArray[np.float64],
        divergence: csr_matrix,
        inverse_symbol: npt.NDArray[np.float64],
        devices: tuple[int, int],
        split: int,
        jit_options: dict | None,
        profile: bool,
    ) -> None:
        first, second = (cp.cuda.Device(i) for i in devices)
        self.devices = (first, second)
        peer = _peer_access(self.devices)
        self.transport = "peer" if peer else "pinned"
        self._momentum = momentum
        self.shape = pressure_dofs.shape
        self._velocity_size = velocity_dofs.size
        plane = math.prod(velocity_dofs.shape[1:]) * velocity_dofs.block_size
        layouts = _layouts(state, velocity_dofs, split)
        a, b = (
            _Slab(device, layout, momentum, state, mass, divergence, self.shape, plane, jit_options)
            for device, layout in zip(self.devices, layouts, strict=True)
        )
        self.slabs = (a, b)
        for own, other in ((a, b), (b, a)):
            with own.device, own.stream:
                rows = other.pressure_order[other.own_pressure :] - own.pressure_rows.start
                if np.any(rows < 0) or np.any(rows >= own.own_pressure):
                    raise ValueError("Adjacent pressure rows fall outside the owning slab.")
                own.remote_rows = cp.asarray(rows)
                own.remote_divergence = cp.empty(len(rows), dtype=np.float64)
        self._transfers = {
            "velocity_halo": tuple(
                _Transfer(src, dst, plane, peer, profile) for src, dst in ((a, b), (b, a))
            ),
            "residual_halo": tuple(
                _Transfer(src, dst, plane, peer, profile) for src, dst in ((a, b), (b, a))
            ),
            "divergence_halo": tuple(
                _Transfer(src, dst, src.ghost_pressure, peer, profile)
                for src, dst in ((a, b), (b, a))
            ),
            "divergence_gather": (_Transfer(b, a, b.own_pressure, peer, profile),),
            "pressure_scatter": (_Transfer(a, b, len(b.pressure_order), peer, profile),),
        }
        with a.device, a.stream:
            self._inverse_symbol = cp.asarray(inverse_symbol)
            self._divergence = cp.empty(math.prod(self.shape), dtype=np.float64)
            self._peer_pressure_rows = cp.asarray(b.pressure_order)
        self._bytes = {
            name: sum(link.nbytes for link in links) for name, links in self._transfers.items()
        }
        self.synchronize()

    def synchronize(self) -> None:
        """Wait for queued operations on both CUDA devices."""
        a, b = self.slabs
        try:
            with a.device:
                a.stream.synchronize()
        finally:
            with b.device:
                b.stream.synchronize()

    def _array_parts(self, velocity: _Parts) -> _Parts:
        if not isinstance(velocity, tuple) or len(velocity) != len(self.slabs):
            raise ValueError("Use distribute() to create a velocity vector for two CUDA devices.")
        for slab, part in zip(self.slabs, velocity, strict=True):
            if (
                not isinstance(part, cp.ndarray)
                or part.shape != (slab.owned,)
                or part.dtype != np.float64
                or part.device.id != slab.device.id
                or not part.flags.c_contiguous
            ):
                raise ValueError("Expected contiguous FP64 slab vectors on their owning devices.")
        return velocity

    def distribute(self, velocity: npt.NDArray[np.float64]) -> tuple[cp.ndarray, cp.ndarray]:
        """Upload velocities in periodic Cartesian DOF order."""
        if (
            not isinstance(velocity, np.ndarray)
            or velocity.shape != (self._velocity_size,)
            or velocity.dtype != np.float64
        ):
            raise ValueError("Expected a flat FP64 periodic host velocity vector.")
        parts = []
        for slab in self.slabs:
            with slab.device, slab.stream:
                parts.append(cp.asarray(velocity[slab.layout.owned], blocking=True))
        self.synchronize()
        return parts[0], parts[1]

    def gather(self, velocity: tuple[cp.ndarray, cp.ndarray]) -> npt.NDArray[np.float64]:
        """Download a velocity vector in Cartesian DOF order."""
        with self.execution(velocity) as (a, b):
            first, second = self.slabs
            return np.concatenate((a.get(stream=first.stream), b.get(stream=second.stream)))

    def constants(self) -> _Parts:
        """Snapshot the current packed form constants on both devices."""
        packed = pack_constants(self._momentum)
        copies = []
        for slab in self.slabs:
            with slab.device, slab.stream:
                copies.append(cp.asarray(packed))
        return copies[0], copies[1]

    @contextmanager
    def execution(self, velocity: _Parts) -> Iterator[_Parts]:
        """Order input before execution and finish queued work."""
        parts = self._array_parts(velocity)
        try:
            for slab in self.slabs:
                with slab.device:
                    current = cp.cuda.get_current_stream()
                    if current != slab.stream:
                        slab.stream.wait_event(current.record())
            yield parts
        finally:
            # Pinned buffers and temporary CUDA arrays are owned by this
            # solver and must outlive each queued copy, even on errors.
            self.synchronize()

    def _residual(self, velocity: _Parts, constants: _Parts) -> _Parts:
        a, b = self.slabs
        for slab, part in zip(self.slabs, velocity, strict=True):
            with slab.device, slab.stream:
                slab.extended[: slab.owned] = part
        s_ab, s_ba = self._transfers["velocity_halo"]
        _exchange(
            (
                (s_ab, velocity[0][: a.plane], b.extended[b.owned :]),
                (s_ba, velocity[1][: b.plane], a.extended[a.owned :]),
            )
        )
        load = []
        for slab, packed in zip(self.slabs, constants, strict=True):
            with slab.device, slab.stream:
                load.append(slab.assembler.assemble((slab.extended,), constants=packed))
        l_ab, l_ba = self._transfers["residual_halo"]
        _exchange(
            (
                (l_ab, load[0][a.owned :], b.remote_load),
                (l_ba, load[1][b.owned :], a.remote_load),
            )
        )
        for slab, result in zip(self.slabs, load, strict=True):
            with slab.device, slab.stream:
                result[: slab.plane] += slab.remote_load
        return load[0][: a.owned], load[1][: b.owned]

    def _assemble_divergence(self, velocity: _Parts) -> None:
        a, b = self.slabs
        partial = []
        for slab, part in zip(self.slabs, velocity, strict=True):
            with slab.device, slab.stream:
                partial.append(slab.divergence @ part)
        d_ab, d_ba = self._transfers["divergence_halo"]
        _exchange(
            (
                (d_ab, partial[0][a.own_pressure :], b.remote_divergence),
                (d_ba, partial[1][b.own_pressure :], a.remote_divergence),
            )
        )
        for slab, div in zip(self.slabs, partial, strict=True):
            with slab.device, slab.stream:
                div[slab.remote_rows] += slab.remote_divergence
        _exchange(
            (
                (
                    self._transfers["divergence_gather"][0],
                    partial[1][: b.own_pressure],
                    self._divergence[b.pressure_rows],
                ),
            )
        )
        with a.device, a.stream:
            self._divergence[a.pressure_rows] = partial[0][: a.own_pressure]

    def _correction_pressure(self, velocity: _Parts, *, scatter: bool) -> cp.ndarray:
        a, b = self.slabs
        self._assemble_divergence(velocity)
        with a.device, a.stream:
            potential = _solve_pressure(self._divergence, self._inverse_symbol, self.shape, cp)
            if scatter:
                a.pressure[:] = potential[a.pressure_indices]
                remote = potential[self._peer_pressure_rows]
        if scatter:
            _exchange(((self._transfers["pressure_scatter"][0], remote, b.pressure),))
        return potential

    def _project(self, velocity: _Parts) -> _Parts:
        self._correction_pressure(velocity, scatter=True)
        results = []
        for slab, part in zip(self.slabs, velocity, strict=True):
            with slab.device, slab.stream:
                results.append(part - slab.inverse_mass * (slab.gradient @ slab.pressure))
        return results[0], results[1]

    def _acceleration(self, velocity: _Parts, constants: _Parts) -> _Parts:
        load = self._residual(velocity, constants)
        result = []
        for slab, part in zip(self.slabs, load, strict=True):
            with slab.device, slab.stream:
                result.append(-slab.inverse_mass * part)
        return result[0], result[1]

    def linear_combination(self, operation: Callable, *vectors: _Parts) -> _Parts:
        """Apply a vector operation to each owned slab."""
        result = []
        for slab, values in zip(self.slabs, zip(*vectors, strict=True), strict=True):
            with slab.device, slab.stream:
                result.append(operation(*values))
        return result[0], result[1]

    def residual(self, velocity: _Parts) -> _Parts:
        """Assemble one distributed load."""
        with self.execution(velocity) as parts:
            return self._residual(parts, self.constants())

    def project(self, velocity: _Parts) -> _Parts:
        """Project a distributed velocity."""
        with self.execution(velocity) as parts:
            return self._project(parts)

    def pressure(self, velocity: _Parts) -> cp.ndarray:
        """Return pressure on the first CUDA device."""
        with self.execution(velocity) as parts:
            acceleration = self._acceleration(parts, self.constants())
            potential = self._correction_pressure(acceleration, scatter=False)
            with self.slabs[0].device, self.slabs[0].stream:
                return -potential

    def weak_divergence(self, velocity: _Parts) -> cp.ndarray:
        """Return the assembled divergence on the first device."""
        with self.execution(velocity) as parts:
            self._assemble_divergence(parts)
            first = self.slabs[0]
            with first.device, first.stream:
                return self._divergence.copy()

    def derivative(self, velocity: _Parts, constants: _Parts) -> _Parts:
        """Project the negative mass-weighted load."""
        return self._project(self._acceleration(velocity, constants))

    def communication_profile(self) -> dict:
        """Collect and clear recorded CUDA timings."""
        return {
            "transport": self.transport,
            "bytes_per_stage": dict(self._bytes),
            "cells_per_device": [len(slab.layout.cells) for slab in self.slabs],
            "seconds": {
                name: [transfer.timings() for transfer in transfers]
                for name, transfers in self._transfers.items()
            },
        }
