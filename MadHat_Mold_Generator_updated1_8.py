"""
MAD HAT — Negative Mold Generator (Python desktop edition)

Install:
    py -m pip install PySide6 pyvista pyvistaqt trimesh manifold3d numpy scipy shapely

Run:
    py MadHat_Negative_Mold_Generator.py

STL input is expected to be Z-up. Generated STL exports are Z-up.
Boolean operations use the Manifold3D engine via trimesh.
"""
from __future__ import annotations

import math
import os
import sys
import zipfile
import json
import io
import traceback
from dataclasses import dataclass

import numpy as np
import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox,
    QComboBox, QSlider, QCheckBox, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QSplitter, QScrollArea, QFrame, QTabWidget
)
try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "Missing 3D preview packages. Install with:\n"
        "py -m pip install PySide6 pyvista pyvistaqt trimesh manifold3d numpy scipy shapely"
    ) from exc


GOLD = "#f0a500"
PALETTE = ["#e34b91", "#9b58e8", "#d5ae45", "#73c95d",
           "#39b8a9", "#8793a8", "#e07848", "#6b9fe0"]


def box_mesh(extents, center):
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def vertical_cylinder(radius, height, sections=32):
    """Cylinder whose axis is the application's Y-up vertical axis."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2, [1,0,0]))
    return mesh


def cylinder_between(radius, depth, start, direction, sections=32):
    """Cylinder centred halfway along a vector."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    mid = np.asarray(start, dtype=float) + direction * depth / 2
    mesh = trimesh.creation.cylinder(radius=radius, height=depth, sections=sections)
    T = trimesh.geometry.align_vectors([0, 0, 1], direction)
    if T is not None:
        mesh.apply_transform(T)
    mesh.apply_translation(mid)
    return mesh


def boolean(meshes, operation):
    if not meshes:
        raise ValueError("No meshes supplied to Boolean operation.")
    if operation == "union":
        result = trimesh.boolean.union(meshes, engine="manifold")
    elif operation == "difference":
        result = trimesh.boolean.difference(meshes, engine="manifold")
    elif operation == "intersection":
        result = trimesh.boolean.intersection(meshes, engine="manifold")
    else:
        raise ValueError(operation)
    if result is None:
        raise RuntimeError("Manifold Boolean failed. Check that the STL is watertight.")
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))
    result.remove_unreferenced_vertices()
    return result


def radial_profile(mesh, height, angle_count=144, y_count=48):
    """Conservative radial envelope of the model at sampled heights/angles."""
    v = mesh.vertices
    radii = np.full((y_count + 1, angle_count), np.nan)
    yi = np.clip(np.rint(v[:, 1] / max(height, 1e-8) * y_count).astype(int), 0, y_count)
    angles = (np.arctan2(v[:, 0], v[:, 2]) + 2 * np.pi) % (2 * np.pi)
    ai = np.floor(angles / (2 * np.pi) * angle_count).astype(int) % angle_count
    rr = np.hypot(v[:, 0], v[:, 2])
    for row, col, radius in zip(yi, ai, rr):
        if not np.isfinite(radii[row, col]) or radius > radii[row, col]:
            radii[row, col] = radius

    # Fill sparse bins using a conservative neighbourhood maximum.
    for _ in range(3):
        old = radii.copy()
        for y in range(y_count + 1):
            for a in range(angle_count):
                if np.isfinite(old[y, a]):
                    continue
                vals = []
                for dy in range(-3, 4):
                    yy = y + dy
                    if 0 <= yy <= y_count:
                        for da in range(-4, 5):
                            val = old[yy, (a + da) % angle_count]
                            if np.isfinite(val):
                                vals.append(val)
                if vals:
                    radii[y, a] = max(vals)

    fallback = max(mesh.extents[0], mesh.extents[2]) / 2
    radii[~np.isfinite(radii)] = fallback
    # Conservative smoothing; keep the envelope outside measured vertices.
    smooth = radii.copy()
    for y in range(1, y_count):
        for a in range(angle_count):
            avg = (radii[y, a] * 4 + radii[y, (a-1) % angle_count] +
                   radii[y, (a+1) % angle_count] + radii[y-1, a] +
                   radii[y+1, a]) / 8
            smooth[y, a] = max(radii[y, a], avg)
    pad = max(1.0, min(2.0, max(mesh.extents[0], mesh.extents[2]) * .004))
    smooth += pad

    # Smooth only the outer wall profile.
    outer = smooth.copy()
    kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16
    for _ in range(8):
        tmp = np.empty_like(outer)
        for y in range(y_count + 1):
            tmp[y] = sum(kernel[k+2] * np.roll(outer[y], k) for k in range(-2, 3))
        new = np.empty_like(tmp)
        for y in range(y_count + 1):
            new[y] = sum(kernel[k+2] * tmp[np.clip(y+k, 0, y_count)] for k in range(-2, 3))
        outer = new
    first = int(math.ceil(.90 * y_count))
    opening = float(np.max(smooth[first]))
    smooth[first:] = opening
    return {"inner": smooth, "outer": outer, "height": height,
            "angles": angle_count, "rows": y_count}


def profile_radius(profile, angle, y, outer=False):
    rows = profile["outer"] if outer else profile["inner"]
    n, ny = profile["angles"], profile["rows"]
    af = ((angle % (2*np.pi)) / (2*np.pi)) * n
    a0 = int(math.floor(af)) % n
    a1 = (a0 + 1) % n
    at = af - math.floor(af)
    yf = np.clip(y / max(profile["height"], 1e-8) * ny, 0, ny)
    y0 = int(math.floor(yf))
    y1 = min(ny, y0 + 1)
    yt = yf - y0
    r0 = rows[y0, a0] * (1-at) + rows[y0, a1] * at
    r1 = rows[y1, a0] * (1-at) + rows[y1, a1] * at
    return float(r0 * (1-yt) + r1 * yt)


def offset_model(mesh, clearance):
    """Build a robust outward offset using a filled voxel distance field.

    Vertex-normal offsets can fold over on dense, highly curved meshes.  A
    volumetric dilation avoids those self-intersections and gives the Boolean
    engine a single, watertight cutter.  The voxel pitch is capped at 0.5 mm
    for useful detail without creating an excessive memory footprint.
    """
    from scipy import ndimage

    distance = max(0.0, float(clearance)) + 0.15
    if not np.isfinite(distance):
        raise ValueError("Cavity clearance must be a finite number.")

    extents = np.asarray(mesh.extents, dtype=float)
    if np.any(~np.isfinite(extents)) or np.any(extents <= 0):
        raise ValueError("Cannot offset a mesh with invalid dimensions.")

    # Keep the grid fine enough to represent the clearance, while limiting
    # the total number of voxels for larger source models.
    pitch = min(0.5, max(0.35, distance / 6.0))
    estimated_voxels = float(np.prod(np.ceil(extents / pitch + 4)))
    if estimated_voxels > 14_000_000:
        pitch *= (estimated_voxels / 14_000_000) ** (1.0 / 3.0)

    try:
        surface = mesh.voxelized(pitch=pitch)
        solid = surface.fill()
        occupied = np.asarray(solid.matrix, dtype=bool)
        if occupied.size == 0 or not occupied.any():
            raise ValueError("Voxelisation produced an empty model.")

        # Add enough empty space around the source for the complete offset.
        pad = int(math.ceil(distance / pitch)) + 3
        padded = np.pad(occupied, pad, mode="constant", constant_values=False)

        # Distance from every empty voxel centre to the filled source volume.
        # Include half a voxel diagonal to compensate for centre sampling and
        # produce a conservative outward clearance.
        outside_distance = ndimage.distance_transform_edt(~padded, sampling=pitch)
        offset_volume = padded | (outside_distance <= distance + pitch * 0.5)

        out = trimesh.voxel.ops.matrix_to_marching_cubes(
            offset_volume, pitch=pitch
        )
        # matrix_to_marching_cubes returns coordinates relative to the padded
        # array. Restore the source voxel grid's world-space origin.
        grid_origin = np.asarray(surface.transform[:3, 3], dtype=float)
        out.apply_translation(grid_origin - pad * pitch)
        out.remove_unreferenced_vertices()
        out.process(validate=True)
        out.fix_normals()

        if (not out.is_watertight or not out.is_winding_consistent or
                not np.isfinite(out.vertices).all() or out.volume <= 0):
            raise RuntimeError(
                "The volumetric cavity offset did not produce a valid closed solid."
            )
        return out
    except Exception as exc:
        raise RuntimeError(
            f"Could not create a clean {distance:.2f} mm cavity offset: {exc}"
        ) from exc


def annular_panel(profile, a0, a1, y_min, y_max, wall, clearance, base_top):
    na, ny = 192, 128
    verts = []
    faces = []
    grid = (na + 1) * (ny + 1)
    reserve = max(5.0, wall * .55)
    for layer in range(2):
        for j in range(ny + 1):
            y = y_min + (y_max-y_min) * j/ny
            for i in range(na + 1):
                a = a0 + (a1-a0) * i/na
                source_y = y - base_top
                if y <= base_top + 1e-6:
                    ri = profile_radius(profile, a, 0) + clearance
                else:
                    ri = .25
                ro = profile_radius(profile, a, source_y, True) + clearance + wall + reserve
                r = ri if layer == 0 else ro
                verts.append([math.sin(a)*max(.2, r), y, math.cos(a)*max(.2, r)])

    def idx(layer, i, j):
        return layer*grid + j*(na+1)+i

    def quad(a,b,c,d):
        faces.extend([[a,b,d],[b,c,d]])

    for layer in range(2):
        for j in range(ny):
            for i in range(na):
                a,b,c,d = idx(layer,i,j),idx(layer,i+1,j),idx(layer,i+1,j+1),idx(layer,i,j+1)
                if layer == 0:
                    quad(a,d,c,b)
                else:
                    quad(a,b,c,d)
    for j in range(ny):
        quad(idx(0,0,j),idx(1,0,j),idx(1,0,j+1),idx(0,0,j+1))
        quad(idx(0,na,j),idx(0,na,j+1),idx(1,na,j+1),idx(1,na,j))
    for i in range(na):
        quad(idx(0,i,0),idx(1,i,0),idx(1,i+1,0),idx(0,i+1,0))
        quad(idx(0,i,ny),idx(1,i,ny),idx(1,i+1,ny),idx(0,i+1,ny))
    result = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
    result.fix_normals()
    return result


def contoured_clip_flange(profile, angle, y_min, y_max, base_top,
                          clearance=0.0, wall=8.0, projection=12.0,
                          thickness=3.0, tangential_offset=0.0,
                          segments=64):
    """
    Build the same height-sampled, contour-following clip flange as the HTML
    generator. The flange root penetrates the panel's outer skin so the union
    is reliable, while its outside face follows the smoothed wall profile.
    """
    verts = []
    faces = []
    tangent_min = tangential_offset - thickness / 2
    tangent_max = tangential_offset + thickness / 2
    outer_wall_reserve = max(5.0, wall * 0.55)
    root_overlap = float(np.clip(wall * 0.2, 1.5, 3.0))

    for j in range(segments + 1):
        y = y_min + (y_max - y_min) * j / segments
        source_y = float(np.clip(y - base_top, 0, profile["height"]))
        panel_outer = (profile_radius(profile, angle, source_y, True)
                       + max(0.0, clearance) + wall + outer_wall_reserve)
        inner_radius = panel_outer - root_overlap
        outer_radius = panel_outer + max(0.0, projection)
        for radius, tangent in (
            (inner_radius, tangent_min),
            (outer_radius, tangent_min),
            (outer_radius, tangent_max),
            (inner_radius, tangent_max),
        ):
            verts.append([
                math.sin(angle) * radius + math.cos(angle) * tangent,
                y,
                math.cos(angle) * radius - math.sin(angle) * tangent,
            ])

    def ring(j, k):
        return j * 4 + k

    for j in range(segments):
        for k in range(4):
            nxt = (k + 1) % 4
            a, b = ring(j, k), ring(j, nxt)
            c, d = ring(j + 1, nxt), ring(j + 1, k)
            faces.extend([[a, b, d], [b, c, d]])

    faces.extend([
        [ring(0, 0), ring(0, 2), ring(0, 1)],
        [ring(0, 0), ring(0, 3), ring(0, 2)],
    ])
    top = ring(segments, 0)
    faces.extend([[top, top + 1, top + 2], [top, top + 2, top + 3]])

    flange = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    flange.fix_normals()
    return flange


def radial_prism(profile, y0, y1, clearance, subtract=0.2, count=192):
    verts, faces = [], []
    for y in (y0, y1):
        for i in range(count):
            a = i/count*2*np.pi
            r = max(1, profile_radius(profile, a, 0) + clearance - subtract)
            verts.append([math.sin(a)*r, y, math.cos(a)*r])
    bottom_center = len(verts); verts.append([0,y0,0])
    top_center = len(verts); verts.append([0,y1,0])
    for i in range(count):
        n=(i+1)%count
        faces.extend([[i,n,count+n],[i,count+n,count+i],
                      [bottom_center,n,i],[top_center,count+i,count+n]])
    out=trimesh.Trimesh(vertices=np.asarray(verts),faces=np.asarray(faces),process=True)
    out.fix_normals()
    return out


def transform_to_export(mesh):
    """Application Y-up to printer STL Z-up, keeping the bottom at Z=0."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2, [1,0,0]))
    out.apply_translation([0, 0, -out.bounds[0,2]])
    return out


def transform_to_preview(mesh):
    """Rotate a preview copy to Z-up while preserving its world position.

    Unlike STL export, the visualiser must not rebase each individual mesh to
    Z=0.  Doing so discards the source model's base_top/lift_y placement and
    makes the source appear to sink into or intersect the mould base.
    """
    out = mesh.copy()
    out.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    )
    return out


@dataclass
class MoldPart:
    name: str
    mesh: trimesh.Trimesh
    direction: tuple = (0.0, 0.0, 0.0)


class MoldGenerator:
    def __init__(self):
        self.source = None
        self.plug = None
        self.parts: list[MoldPart] = []
        self.profile = None
        self.lift_y = 0.0
        self.base_top = 0.0
        self.cavity_volume_mm3 = 0.0
        self.pour_channel_volume_mm3 = 0.0
        self.total_resin_volume_mm3 = 0.0

    def load_stl(self, path):
        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("The selected STL does not contain a triangle mesh.")
        # STL is Z-up; rotate into the generator's Y-up coordinates.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2, [1,0,0]))
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        mesh.apply_translation([-mesh.bounds.mean(axis=0)[0], -mesh.bounds[0,1],
                                -mesh.bounds.mean(axis=0)[2]])
        if not mesh.is_watertight:
            raise ValueError("The uploaded STL is not watertight. Repair it in a mesh tool and retry.")
        self.source = mesh
        self.parts.clear()
        return mesh.extents

    def generate(self, panels=4, top_extra=8, wall=8, base=6, clearance=3,
                 fill_diameter=12, fill_depth=20, progress=None):
        if self.source is None:
            raise ValueError("Load an STL first.")
        src = self.source.copy()
        size = src.extents
        if np.any(size <= 0):
            raise ValueError("The STL has invalid or zero dimensions.")
        wall=max(3,float(wall)); base=max(3,float(base))
        y_min=1-base
        base_bottom=y_min
        actual_base=max(1,base-.5)
        base_top=base_bottom+actual_base
        self.base_top = base_top
        y_max=base_top+size[1]+max(0,float(top_extra))
        profile=radial_profile(src,size[1])
        self.profile=profile
        cutter=offset_model(src,clearance)
        # The offset cutter encloses the expanded model envelope. Resin fills
        # only the gap between that envelope and the original STL, so subtract
        # the original solid volume from the offset volume.
        expanded_volume = abs(float(cutter.volume))
        source_volume = abs(float(src.volume))
        self.cavity_volume_mm3 = expanded_volume - source_volume
        if (not np.isfinite(self.cavity_volume_mm3) or
                self.cavity_volume_mm3 <= 0):
            raise ValueError(
                "Could not calculate a valid silicone gap volume. "
                "Check that the source STL is a closed, watertight solid "
                "and that the cavity clearance is greater than zero."
            )
        cutter.apply_translation([0,base_top,0])

        # Calculate the pour channel's additional volume. The channel is
        # counted only where it lies outside the expanded model envelope,
        # preventing overlap with the already-counted resin gap.
        hole_diameter = max(2.0, min(50.0, float(fill_diameter)))
        hole_depth = max(1.0, min(float(fill_depth), y_max - y_min))
        pour_channel = vertical_cylinder(
            radius=hole_diameter / 2,
            height=hole_depth,
            sections=96,
        )
        # Measure only the portion inside the mold, not the extra 1 mm
        # cutter overrun used by the panel Boolean operation.
        pour_channel.apply_translation(
            [0, y_max - hole_depth / 2, 0]
        )
        channel_overlap = boolean(
            [pour_channel, cutter], "intersection"
        )
        channel_overlap_volume = (
            abs(float(channel_overlap.volume))
            if channel_overlap is not None and len(channel_overlap.faces)
            else 0.0
        )
        self.pour_channel_volume_mm3 = max(
            0.0, abs(float(pour_channel.volume)) - channel_overlap_volume
        )
        self.total_resin_volume_mm3 = (
            self.cavity_volume_mm3 + self.pour_channel_volume_mm3
        )

        if progress: progress("Building fitted base…")
        base_blank=radial_prism(profile,base_bottom,base_top,clearance)
        parts=[]
        # Base + locating pins.
        pin_offset=max(0,size[0]*.22)
        for x in (-pin_offset,pin_offset):
            pin=vertical_cylinder(radius=1.9,height=3.2,sections=32)
            pin.apply_translation([x,base_top+1.4,0])
            base_blank=boolean([base_blank,pin],"union")
        base_mesh=base_blank

        for i in range(int(panels)):
            if progress: progress(f"Generating cavity panel {i+1}/{panels}…")
            a0=i/panels*2*np.pi; a1=(i+1)/panels*2*np.pi
            blank=annular_panel(profile,a0,a1,y_min,y_max,wall,clearance,base_top)
            panel=boolean([blank,cutter],"difference")
            # Flange-mounted alignment nubs below provide panel registration.
            # Contoured external clip flanges. Their full height follows the
            # smoothed outer wall profile instead of using a rectangular block.
            # Opposing tabs are shifted tangentially by half their thickness,
            # matching the HTML generator's edge-to-edge arrangement.
            flange_projection = 12.0
            flange_thickness = 3.0
            flange_offset = flange_thickness / 2
            # Each seam uses a male/female pair across the touching side
            # faces of the offset flanges.  The two locating nubs sit near the
            # top and bottom of the flange and are centred across its radial
            # width (the middle of the flange's projection).
            flange_nub_radius = 1.5
            flange_hole_radius = 1.8
            flange_nub_depth = 2.8
            flange_end_inset = min(
                (y_max - y_min) * 0.2,
                max(4.0, flange_nub_radius + 1.5)
            )
            flange_nub_heights = (
                y_min + flange_end_inset,
                y_max - flange_end_inset,
            )

            for edge, angle in ((0, a0), (1, a1)):
                tangential_offset = flange_offset if edge == 0 else -flange_offset
                flange = contoured_clip_flange(
                    profile, angle, y_min, y_max, base_top,
                    clearance=clearance, wall=wall,
                    projection=flange_projection,
                    thickness=flange_thickness,
                    tangential_offset=tangential_offset,
                )

                # At every seam, the right-hand flange (edge 1) carries
                # outward-facing male nubs.  The next panel's left-hand
                # flange (edge 0) receives matching blind female sockets.
                tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)])
                radial = np.array([math.sin(angle), 0.0, math.cos(angle)])
                side_tangent = (
                    tangential_offset + flange_thickness / 2
                    if edge == 1 else
                    tangential_offset - flange_thickness / 2
                )

                for nub_y in flange_nub_heights:
                    source_y = float(np.clip(nub_y - base_top, 0, profile["height"]))
                    panel_outer = (
                        profile_radius(profile, angle, source_y, True)
                        + max(0.0, clearance) + wall
                        + max(5.0, wall * 0.55)
                    )
                    nub_radial = panel_outer + flange_projection / 2
                    face_point = (
                        radial * nub_radial
                        + np.array([0.0, nub_y, 0.0])
                        + tangent * side_tangent
                    )

                    if edge == 1:
                        # Start slightly inside the flange face to guarantee
                        # a sound union, then project into the mating flange.
                        male_start = face_point - tangent * 0.35
                        male = cylinder_between(
                            flange_nub_radius, flange_nub_depth,
                            male_start, tangent
                        )
                        flange = boolean([flange, male], "union")
                    else:
                        # Matching socket, with a small radial clearance.
                        socket_start = face_point - tangent * 0.25
                        socket = cylinder_between(
                            flange_hole_radius, flange_nub_depth + 0.5,
                            socket_start, tangent
                        )
                        flange = boolean([flange, socket], "difference")

                panel = boolean([panel, flange], "union")

            # Cut the adjustable silicone pour channel vertically through the
            # centre of the assembled mold, matching the HTML implementation.
            # trimesh cylinders are Z-axis by default, so use the Y-axis helper.
            hole_diameter = max(2.0, min(50.0, float(fill_diameter)))
            hole_depth = max(1.0, min(float(fill_depth), y_max - y_min))
            hole = vertical_cylinder(
                radius=hole_diameter / 2,
                height=hole_depth + 1.0,
                sections=64,
            )
            hole.apply_translation([0, y_max - hole_depth / 2 + 0.5, 0])
            panel = boolean([panel, hole], "difference")
            parts.append(MoldPart(f"Mold_Panel_{i+1:02d}",panel,
                                  (math.sin((i+.5)/panels*2*np.pi),0,
                                   math.cos((i+.5)/panels*2*np.pi))))

        # Horizontal base-to-wall locating stubs and matching blind sockets.
        for i in range(int(panels)):
            angle=(i+.5)/panels*2*np.pi
            inner=profile_radius(profile,angle,0)+clearance
            nub_radius=inner-.1
            point=np.array([math.sin(angle)*nub_radius, y_min+2.5,
                            math.cos(angle)*nub_radius])
            radial=np.array([math.sin(angle),0,math.cos(angle)])
            nub=cylinder_between(1.0,2.0,point-radial,radial)
            base_mesh=boolean([base_mesh,nub],"union")
            socket=cylinder_between(1.25,4.4,point-radial*2.2,radial)
            idx=i
            parts[idx].mesh=boolean([parts[idx].mesh,socket],"difference")

        # Add matching blind holes to a copy of the original plug.
        plug=src.copy()
        for x in (-pin_offset,pin_offset):
            hole=vertical_cylinder(radius=2.25,height=3.8,sections=32)
            hole.apply_translation([x,1.8,0])
            plug=boolean([plug,hole],"difference")
        self.plug=plug

        # Align all parts to the build plate.
        all_meshes=[base_mesh]+[p.mesh for p in parts]
        lowest=min(m.bounds[0,1] for m in all_meshes)
        self.lift_y=-.2-lowest
        for m in all_meshes:
            m.apply_translation([0,self.lift_y,0])
        self.parts=[MoldPart("Mold_Base",base_mesh)]+parts
        return self.parts

    def generate_flat_2d(self, wall=8.0, base=6.0, top_extra=8.0, margin=10.0,
                         progress=None):
        """Create a contour-following flat pour mould around the STL footprint.

        The well, perimeter wall and base follow the projected X/Z silhouette
        of the source model instead of using a rectangular box.
        """
        if self.source is None:
            raise ValueError("Load an STL first.")
        from shapely.geometry import Polygon
        from shapely.ops import unary_union, triangulate

        src = self.source.copy()
        sx, sy, sz = map(float, src.extents)
        if min(sx, sy, sz) <= 0:
            raise ValueError("The STL has invalid or zero dimensions.")

        wall = max(2.0, float(wall))
        base = max(2.0, float(base))
        margin = max(2.0, float(margin))
        wall_height = sy + max(0.0, float(top_extra))

        if progress:
            progress("Tracing STL silhouette and building contoured mould…")

        # Project every surface triangle onto the horizontal X/Z plane.
        # The union of these projected triangles describes the model's
        # actual footprint, including concave outlines, rather than its bounds.
        projected = []
        vertices = np.asarray(src.vertices)
        for face in np.asarray(src.faces):
            points = vertices[face][:, [0, 2]]
            poly = Polygon(points)
            if poly.is_valid and poly.area > 1e-7:
                projected.append(poly)
        if not projected:
            raise ValueError("Could not extract a usable 2D outline from this STL.")

        footprint = unary_union(projected).buffer(0)
        if footprint.is_empty:
            raise ValueError("The STL silhouette is empty.")
        # A single continuous pour tray is required. If the projection has
        # separate islands, join nearby islands using the requested margin.
        inner_shape = footprint.buffer(margin, join_style=1).buffer(0)
        if inner_shape.geom_type == "MultiPolygon":
            inner_shape = unary_union(list(inner_shape.geoms)).buffer(0)
        # Use the exterior silhouette for a clean, manufacturable tray wall.
        if inner_shape.geom_type == "MultiPolygon":
            inner_shape = max(inner_shape.geoms, key=lambda g: g.area)
        inner_shape = Polygon(inner_shape.exterior).buffer(0)
        outer_shape = inner_shape.buffer(wall, join_style=1).buffer(0)
        if outer_shape.geom_type == "MultiPolygon":
            outer_shape = max(outer_shape.geoms, key=lambda g: g.area)
        outer_shape = Polygon(outer_shape.exterior).buffer(0)

        def polygon_faces(poly, y, reverse=False):
            """Triangulate a polygon cap, retaining only triangles inside it."""
            points = []
            faces = []
            for tri in triangulate(poly):
                if not poly.covers(tri.representative_point()):
                    continue
                coords = list(tri.exterior.coords)[:3]
                offset = len(points)
                points.extend([[x, y, z] for x, z in coords])
                faces.append([offset, offset + 1, offset + 2])
            if reverse:
                faces = [f[::-1] for f in faces]
            return points, faces

        def extruded_polygon(poly, y0, y1):
            """Watertight vertical extrusion of a polygon without holes."""
            poly = Polygon(poly.exterior)
            ring = list(poly.exterior.coords)[:-1]
            n = len(ring)
            verts = [[x, y0, z] for x, z in ring]
            verts += [[x, y1, z] for x, z in ring]
            faces = []
            p0, f0 = polygon_faces(poly, y0, reverse=True)
            p1, f1 = polygon_faces(poly, y1, reverse=False)
            # Cap triangulation uses independent vertices; append those caps.
            base0 = len(verts)
            verts.extend(p0)
            faces.extend([[base0 + a, base0 + b, base0 + c] for a,b,c in f0])
            base1 = len(verts)
            verts.extend(p1)
            faces.extend([[base1 + a, base1 + b, base1 + c] for a,b,c in f1])
            for i in range(n):
                j = (i + 1) % n
                faces.extend([[i, j, n + j], [i, n + j, n + i]])
            mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                                   faces=np.asarray(faces, dtype=np.int64),
                                   process=True)
            mesh.fix_normals()
            return mesh

        def wall_ring_mesh(inner, outer, y0, y1):
            """Build a closed extruded ring: top/bottom annuli and both sides."""
            ring_poly = outer.difference(inner).buffer(0)
            if ring_poly.geom_type == "MultiPolygon":
                ring_poly = unary_union(list(ring_poly.geoms))
            verts, faces = [], []
            # Triangulated annular caps.
            for y, reverse in ((y0, True), (y1, False)):
                for tri in triangulate(ring_poly):
                    if not ring_poly.covers(tri.representative_point()):
                        continue
                    coords = list(tri.exterior.coords)[:3]
                    base_idx = len(verts)
                    verts.extend([[x, y, z] for x, z in coords])
                    face = [base_idx, base_idx+1, base_idx+2]
                    faces.append(face[::-1] if reverse else face)
            # Vertical faces on every exterior and interior boundary.
            polygons = list(ring_poly.geoms) if ring_poly.geom_type == "MultiPolygon" else [ring_poly]
            for component in polygons:
                for boundary in [component.exterior, *component.interiors]:
                    coords = list(boundary.coords)
                    for a, b in zip(coords[:-1], coords[1:]):
                        idx = len(verts)
                        verts.extend([[a[0], y0, a[1]], [b[0], y0, b[1]],
                                      [b[0], y1, b[1]], [a[0], y1, a[1]]])
                        faces.extend([[idx, idx+1, idx+2], [idx, idx+2, idx+3]])
            mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                                   faces=np.asarray(faces, dtype=np.int64),
                                   process=True)
            mesh.fix_normals()
            return mesh

        if progress:
            progress("Forming contoured base and perimeter wall…")
        base_mesh = extruded_polygon(outer_shape, -base + 0.2, 0.0)
        walls_mesh = wall_ring_mesh(inner_shape, outer_shape, 0.0, wall_height)
        tray = boolean([base_mesh, walls_mesh, src], "union")
        tray.fix_normals()
        if not tray.is_watertight:
            raise RuntimeError("The contour-following 2D mould did not produce a watertight solid.")

        # Silicone fills the contoured well around the raised STL plug.
        well_volume = float(inner_shape.area) * wall_height
        self.cavity_volume_mm3 = max(0.0, well_volume - abs(float(src.volume)))
        self.pour_channel_volume_mm3 = 0.0
        self.total_resin_volume_mm3 = self.cavity_volume_mm3
        self.base_top = 0.0
        self.lift_y = 0.0
        self.profile = None
        self.plug = src.copy()
        self.parts = [MoldPart("Flat_2D_Contoured_Negative_Mold", tray)]
        return self.parts

    def export_part(self, index, path):
        transform_to_export(self.parts[index].mesh).export(path, file_type="stl")

    def export_zip(self, path, exploded=0):
        max_size=max(self.source.extents)
        dist=(exploded/100)*(8+max(12,max_size*.12))
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as zf:
            for i,part in enumerate(self.parts):
                mesh=part.mesh.copy()
                if i:
                    mesh.apply_translation(np.array(part.direction)*dist)
                out=transform_to_export(mesh)
                zf.writestr(part.name+".stl",out.export(file_type="stl"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAD HAT · NEGATIVE MOLD GENERATOR")
        self.resize(1500, 900)
        self.generator=MoldGenerator()
        self.source_original_mesh = None
        self.stl_models = []
        self.stl_model_visible = []
        self.explode_value=0
        self.source_actor=None
        self.source_edge_actor=None
        self.part_actors=[]
        self.outline_actors=[]
        self._build_ui()
        self._style()

    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # Left control workspace
        left = QWidget()
        side = QVBoxLayout(left)
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(12)

        header = QFrame()
        header.setObjectName("appHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        title = QLabel("MAD HAT")
        title.setObjectName("appTitle")
        subtitle = QLabel("NEGATIVE MOLD GENERATOR  ·  DESKTOP")
        subtitle.setObjectName("appSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        side.addWidget(header)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        side.addWidget(tabs, 1)

        # --- Build tab: source model, mould settings and silicone estimate ---
        build_page = QWidget()
        build_layout = QVBoxLayout(build_page)
        build_layout.setContentsMargins(8, 12, 8, 8)
        build_layout.setSpacing(12)

        source = QGroupBox("01  ·  SOURCE MODEL")
        sf = QVBoxLayout(source)
        sf.setSpacing(8)
        self.model_info = QLabel("No model loaded.\nUpload an STL to get started.")
        self.model_info.setObjectName("modelInfo")
        self.model_info.setWordWrap(True)
        sf.addWidget(self.model_info)

        source_actions = QHBoxLayout()
        self.load_btn = QPushButton("ADD STL")
        self.load_btn.setObjectName("primaryButton")
        self.load_btn.clicked.connect(self.load_stl)
        source_actions.addWidget(self.load_btn, 2)

        self.delete_stl_btn = QPushButton("DELETE")
        self.delete_stl_btn.setEnabled(False)
        self.delete_stl_btn.clicked.connect(self.delete_stl)
        source_actions.addWidget(self.delete_stl_btn, 1)
        sf.addLayout(source_actions)

        project_actions = QHBoxLayout()
        self.save_project_btn = QPushButton("SAVE PROJECT")
        self.save_project_btn.setEnabled(False)
        self.save_project_btn.clicked.connect(self.save_project)
        project_actions.addWidget(self.save_project_btn)
        self.load_project_btn = QPushButton("OPEN PROJECT")
        self.load_project_btn.clicked.connect(self.load_project)
        project_actions.addWidget(self.load_project_btn)
        sf.addLayout(project_actions)
        build_layout.addWidget(source)

        opts = QGroupBox("02  ·  MOULD CONFIGURATION")
        form = QFormLayout(opts)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.mold_style = QComboBox()
        self.mold_style.addItem("Multi-piece 3D mold", "multi")
        self.mold_style.addItem("2D flat pour mold", "flat")
        form.addRow("Mould type", self.mold_style)

        self.panel_count = QComboBox()
        for n in (2, 4, 5, 6, 8):
            self.panel_count.addItem(f"{n} panels", n)
        self.top_extra = self.spin(8, 0, 100, 1)
        self.wall = self.spin(8, 3, 100, 1)
        self.base = self.spin(6, 3, 100, 1)
        self.clearance = self.spin(3, 0, 20, .1)
        self.fill_dia = self.spin(12, 2, 50, .5)
        self.fill_depth = self.spin(20, 1, 500, 1)

        form.addRow("Removable panels", self.panel_count)
        form.addRow("Top height extra (mm)", self.top_extra)
        form.addRow("Wall thickness (mm)", self.wall)
        form.addRow("Base thickness (mm)", self.base)
        form.addRow("Cavity clearance (mm)", self.clearance)
        form.addRow("Fill hole diameter (mm)", self.fill_dia)
        form.addRow("Fill hole depth (mm)", self.fill_depth)

        self.generate_btn = QPushButton("GENERATE MOLD")
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.setMinimumHeight(42)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generate)
        form.addRow(self.generate_btn)

        self.generation_readout = QLabel("Ready to generate.")
        self.generation_readout.setObjectName("generationReadout")
        self.generation_readout.setWordWrap(True)
        form.addRow(self.generation_readout)

        self.status = QLabel("Upload an STL to begin.")
        self.status.setObjectName("statusLabel")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        build_layout.addWidget(opts)

        stl_resin = QGroupBox("03  ·  RESIN REQUIRED — STL ONLY")
        stl_resin_layout = QVBoxLayout(stl_resin)
        stl_resin_layout.setSpacing(7)
        self.stl_resin_volume = QLabel("Original STL volume: —")
        self.stl_resin_fill = QLabel("Resin to fill STL: —")
        self.stl_resin_mix = QLabel("Mixing amount (+10%): —")
        self.stl_resin_volume.setObjectName("capacityDetail")
        self.stl_resin_fill.setObjectName("capacityTotal")
        self.stl_resin_mix.setObjectName("capacityMix")
        self.stl_resin_note = QLabel(
            "Calculates the volume inside the original watertight STL only. "
            "Does not include the mould cavity, walls or pour channel.")
        self.stl_resin_note.setWordWrap(True)
        self.stl_resin_note.setObjectName("mutedText")
        for label in (self.stl_resin_volume, self.stl_resin_fill,
                      self.stl_resin_mix, self.stl_resin_note):
            stl_resin_layout.addWidget(label)
        build_layout.addWidget(stl_resin)

        capacity = QGroupBox("04  ·  SILICONE REQUIRED")
        capacity_layout = QVBoxLayout(capacity)
        capacity_layout.setSpacing(7)
        self.capacity_volume = QLabel("Silicone gap required: —")
        self.capacity_channel = QLabel("Pour channel volume: —")
        self.capacity_total = QLabel("Total silicone required: —")
        self.capacity_extra = QLabel("Mixing amount (+10%): —")
        self.capacity_volume.setObjectName("capacityDetail")
        self.capacity_channel.setObjectName("capacityDetail")
        self.capacity_total.setObjectName("capacityTotal")
        self.capacity_extra.setObjectName("capacityMix")
        self.capacity_note = QLabel(
            "Includes the silicone gap and pour channel without double-counting "
            "their overlap. Values are shown in mL and litres.")
        self.capacity_note.setWordWrap(True)
        self.capacity_note.setObjectName("mutedText")
        for label in (self.capacity_volume, self.capacity_channel,
                      self.capacity_total, self.capacity_extra, self.capacity_note):
            capacity_layout.addWidget(label)
        build_layout.addWidget(capacity)
        build_layout.addStretch(1)
        tabs.addTab(build_page, "BUILD")

        # --- Preview tab: visibility, exploded view and export tools ---
        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(8, 12, 8, 8)
        preview_layout.setSpacing(12)

        stl_selection_group = QGroupBox("STL SELECTION")
        stl_selection_layout = QVBoxLayout(stl_selection_group)
        stl_selection_layout.setSpacing(6)
        self.stl_selection = QListWidget()
        self.stl_selection.setMaximumHeight(125)
        self.stl_selection.currentRowChanged.connect(self.stl_selection_changed)
        stl_selection_layout.addWidget(self.stl_selection)
        stl_visibility_row = QHBoxLayout()
        self.stl_show_btn = QPushButton("SHOW")
        self.stl_show_btn.clicked.connect(lambda: self.set_selected_stl_visibility(True))
        self.stl_hide_btn = QPushButton("HIDE")
        self.stl_hide_btn.clicked.connect(lambda: self.set_selected_stl_visibility(False))
        self.stl_show_all_btn = QPushButton("SHOW ALL")
        self.stl_show_all_btn.clicked.connect(self.show_all_stls)
        self.stl_delete_selected_btn = QPushButton("DELETE")
        self.stl_delete_selected_btn.clicked.connect(self.delete_selected_stl)
        for button in (self.stl_show_btn, self.stl_hide_btn,
                       self.stl_show_all_btn, self.stl_delete_selected_btn):
            button.setEnabled(False)
            stl_visibility_row.addWidget(button)
        stl_selection_layout.addLayout(stl_visibility_row)
        stl_selection_hint = QLabel("Select an STL here to move, scale, rotate or duplicate it in the Transform tab.")
        stl_selection_hint.setWordWrap(True)
        stl_selection_hint.setObjectName("mutedText")
        stl_selection_layout.addWidget(stl_selection_hint)
        preview_layout.addWidget(stl_selection_group)

        exp = QGroupBox("04  ·  PREVIEW CONTROLS")
        ef = QVBoxLayout(exp)
        ef.setSpacing(9)
        self.show_source = QCheckBox("Show original STL / plug")
        self.show_source.setChecked(True)
        self.show_source.toggled.connect(self.toggle_source)
        ef.addWidget(self.show_source)

        ef.addWidget(QLabel("Explode amount"))
        self.explode = QSlider(Qt.Horizontal)
        self.explode.setRange(0, 100)
        self.explode.valueChanged.connect(self.apply_explode)
        ef.addWidget(self.explode)
        self.explode_label = QLabel("0%")
        self.explode_label.setObjectName("explodeValue")
        ef.addWidget(self.explode_label)

        self.explode_btn = QPushButton("EXPLODE MOULD")
        self.explode_btn.setObjectName("primaryButton")
        self.explode_btn.setEnabled(False)
        self.explode_btn.clicked.connect(self.toggle_exploded_view)
        ef.addWidget(self.explode_btn)

        hint = QLabel("Use the list below to show or hide individual mold parts. "
                      "Hidden parts remain hidden while exploding the assembly.")
        hint.setWordWrap(True)
        hint.setObjectName("mutedText")
        ef.addWidget(hint)
        preview_layout.addWidget(exp)

        parts_group = QGroupBox("PART VISIBILITY")
        parts_layout = QVBoxLayout(parts_group)
        parts_layout.setContentsMargins(10, 18, 10, 10)
        parts_layout.setSpacing(4)
        self.visibility = QListWidget()
        self.visibility.setUniformItemSizes(True)
        self.visibility.setSpacing(1)
        self.visibility.itemChanged.connect(self.visibility_changed)

        # The largest supported assembly is one base plus eight panels.
        # Size the list to show exactly those nine entries without leaving
        # a large empty area when fewer parts (or a flat mould) are generated.
        max_visible_parts = 1 + max(
            self.panel_count.itemData(i)
            for i in range(self.panel_count.count())
        )
        row_height = self.visibility.fontMetrics().height() + 10
        self.visibility.setFixedHeight(max_visible_parts * row_height + 10)

        parts_layout.addWidget(self.visibility)

        # Keep the outer group background/border snug around the resized list.
        # This includes the title area and the layout's inner margins.
        parts_group.setFixedHeight(self.visibility.height() + 18 + 10 + 10 + 12)
        preview_layout.addWidget(parts_group)

        export_group = QGroupBox("EXPORT")
        export_layout = QVBoxLayout(export_group)
        self.export_selected = QPushButton("EXPORT SELECTED PART  ·  STL")
        self.export_selected.clicked.connect(self.export_one)
        export_layout.addWidget(self.export_selected)
        self.export_all = QPushButton("EXPORT ALL PARTS  ·  ZIP")
        self.export_all.clicked.connect(self.export_all_parts)
        export_layout.addWidget(self.export_all)
        self.export_exploded = QPushButton("EXPORT EXPLODED MOULD  ·  ZIP")
        self.export_exploded.clicked.connect(self.export_exploded_parts)
        export_layout.addWidget(self.export_exploded)
        self.export_plug = QPushButton("EXPORT LOCATED ORIGINAL PLUG  ·  STL")
        self.export_plug.clicked.connect(self.export_plug_stl)
        export_layout.addWidget(self.export_plug)
        for button in (self.export_selected, self.export_all,
                       self.export_exploded, self.export_plug):
            button.setEnabled(False)
        preview_layout.addWidget(export_group)
        tabs.addTab(preview_page, "PREVIEW & EXPORT")

        # --- Transform tab: scale and orient the imported STL before moulding ---
        transform_page = QWidget()
        transform_layout = QVBoxLayout(transform_page)
        transform_layout.setContentsMargins(8, 12, 8, 8)
        transform_layout.setSpacing(12)

        scale_group = QGroupBox("01  ·  SCALE MODEL")
        scale_form = QFormLayout(scale_group)
        self.uniform_scale = QCheckBox("Lock X / Y / Z scale together")
        self.uniform_scale.setChecked(True)
        scale_form.addRow(self.uniform_scale)
        self.scale_x = self.spin(1.0, 0.01, 100.0, 0.05)
        self.scale_y = self.spin(1.0, 0.01, 100.0, 0.05)
        self.scale_z = self.spin(1.0, 0.01, 100.0, 0.05)
        self.scale_x.valueChanged.connect(lambda value: self._sync_uniform_scale("x", value))
        self.scale_y.valueChanged.connect(lambda value: self._sync_uniform_scale("y", value))
        self.scale_z.valueChanged.connect(lambda value: self._sync_uniform_scale("z", value))
        for widget in (self.scale_x, self.scale_y, self.scale_z):
            widget.setSuffix(" ×")
        scale_form.addRow("Width scale (X)", self.scale_x)
        scale_form.addRow("Height scale (Y)", self.scale_y)
        scale_form.addRow("Length scale (Z)", self.scale_z)
        transform_layout.addWidget(scale_group)

        rotate_group = QGroupBox("02  ·  ROTATE MODEL")
        rotate_form = QFormLayout(rotate_group)
        self.rotate_x = self.spin(0, -360, 360, 5)
        self.rotate_y = self.spin(0, -360, 360, 5)
        self.rotate_z = self.spin(0, -360, 360, 5)
        for widget in (self.rotate_x, self.rotate_y, self.rotate_z):
            widget.setSuffix("°")
        rotate_form.addRow("Rotate X", self.rotate_x)
        rotate_form.addRow("Rotate Y", self.rotate_y)
        rotate_form.addRow("Rotate Z", self.rotate_z)
        transform_layout.addWidget(rotate_group)

        move_group = QGroupBox("03  ·  MOVE MODEL")
        move_form = QFormLayout(move_group)
        self.move_x = self.spin(0, -1000, 1000, 1)
        self.move_y = self.spin(0, -1000, 1000, 1)
        self.move_z = self.spin(0, -1000, 1000, 1)
        for widget in (self.move_x, self.move_y, self.move_z):
            widget.setSuffix(" mm")
        move_form.addRow("Move X", self.move_x)
        move_form.addRow("Move Y", self.move_y)
        move_form.addRow("Move Z", self.move_z)
        transform_layout.addWidget(move_group)

        duplicate_group = QGroupBox("04  ·  DUPLICATE MODEL")
        duplicate_layout = QVBoxLayout(duplicate_group)
        self.duplicate_btn = QPushButton("DUPLICATE STL")
        self.duplicate_btn.setEnabled(False)
        self.duplicate_btn.clicked.connect(self.duplicate_model)
        duplicate_layout.addWidget(self.duplicate_btn)
        self.duplicate_note = QLabel(
            "Adds another copy of the STL beside the current model. "
            "The duplicated geometry is included when generating the mould.")
        self.duplicate_note.setWordWrap(True)
        self.duplicate_note.setObjectName("mutedText")
        duplicate_layout.addWidget(self.duplicate_note)
        transform_layout.addWidget(duplicate_group)

        transform_note = QLabel(
            "Scale, rotation and move offsets are applied before mould generation. "
            "Duplicate adds a second copy beside the first. Applying changes clears "
            "any previously generated mould so it can be regenerated from the updated geometry.")
        transform_note.setWordWrap(True)
        transform_note.setObjectName("mutedText")
        transform_layout.addWidget(transform_note)

        self.apply_transform_btn = QPushButton("APPLY TRANSFORM")
        self.apply_transform_btn.setObjectName("primaryButton")
        self.apply_transform_btn.setMinimumHeight(42)
        self.apply_transform_btn.setEnabled(False)
        self.apply_transform_btn.clicked.connect(self.apply_model_transform)
        transform_layout.addWidget(self.apply_transform_btn)

        self.reset_transform_btn = QPushButton("RESET TRANSFORM")
        self.reset_transform_btn.setEnabled(False)
        self.reset_transform_btn.clicked.connect(self.reset_model_transform)
        transform_layout.addWidget(self.reset_transform_btn)
        transform_layout.addStretch(1)
        tabs.addTab(transform_page, "SCALE & TRANSFORM")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        side.addWidget(self.progress)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        settings_scroll.setWidget(left)

        # Main 3D viewport
        viewport = QFrame()
        viewport.setObjectName("viewportFrame")
        viewport_layout = QVBoxLayout(viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#111821", top="#263342")
        self.plotter.add_axes(line_width=2, color="#d7dde5")

        # Camera shortcuts stay above the 3D viewport while the scene updates.
        view_bar = QFrame()
        view_bar.setObjectName("viewToolbar")
        view_layout = QHBoxLayout(view_bar)
        view_layout.setContentsMargins(10, 7, 10, 7)
        view_layout.setSpacing(6)
        view_layout.addWidget(QLabel("VIEW"))
        self.view_buttons = {}
        for label in ("TOP", "FRONT", "BACK", "LEFT", "RIGHT", "ISO"):
            button = QPushButton(label)
            button.setObjectName("viewButton")
            button.setCheckable(True)
            button.setChecked(label == "ISO")
            button.setMinimumWidth(64)
            button.clicked.connect(
                lambda checked=False, view=label.lower():
                    self.set_visualiser_view(view)
            )
            self.view_buttons[label] = button
            view_layout.addWidget(button)
        view_layout.addStretch(1)
        viewport_layout.addWidget(view_bar)

        try:
            self.plotter.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        viewport_layout.addWidget(self.plotter.interactor)

        split.addWidget(settings_scroll)
        split.addWidget(viewport)
        split.setSizes([430, 1070])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        root_layout.addWidget(split)
        self.setCentralWidget(root)

    @staticmethod
    def spin(value, minimum, maximum, step):
        w=QDoubleSpinBox(); w.setRange(minimum,maximum); w.setSingleStep(step); w.setValue(value)
        w.setDecimals(2 if step<1 else 0)
        return w

    def _style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #151a20; color: #e8edf2; font: 13px "Segoe UI";
            }
            QFrame#appHeader {
                background: #202832; border: 1px solid #35414d;
                border-radius: 10px;
            }
            QLabel#appTitle {
                color: #f0a500; font-size: 23px; font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#appSubtitle {
                color: #9eabb8; font-size: 10px; font-weight: 700;
                letter-spacing: 1.4px;
            }
            QTabWidget::pane {
                border: 1px solid #303b46; border-radius: 8px;
                background: #181f27; top: -1px;
            }
            QTabBar::tab {
                background: #222b35; color: #aeb9c4; padding: 10px 18px;
                margin-right: 4px; border-top-left-radius: 6px;
                border-top-right-radius: 6px; font-weight: 700;
            }
            QTabBar::tab:selected {
                color: #f0a500; background: #303b46;
                border-bottom: 2px solid #f0a500;
            }
            QGroupBox {
                background: #1d252e; border: 1px solid #35414d;
                border-radius: 8px; margin-top: 12px; padding: 12px;
                font-weight: 700; color: #f0a500;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 11px; padding: 0 5px;
            }
            QLabel#modelInfo {
                background: #151c23; border: 1px solid #303b46;
                border-radius: 6px; padding: 10px; color: #c4ced8;
            }
            QLabel#generationReadout {
                color: #f0a500; font-size: 11px; font-weight: 700;
                background: #151c23; border: 1px solid #35414d;
                border-radius: 6px; padding: 9px;
            }
            QLabel#statusLabel {
                color: #aeb9c4; font-size: 11px; padding: 6px 2px;
            }
            QLabel#mutedText { color: #8f9ba7; font-size: 11px; }
            QLabel#capacityDetail { color: #d3dce5; padding: 2px; }
            QLabel#capacityTotal {
                color: #f0a500; font-size: 15px; font-weight: 800;
                padding: 5px 2px;
            }
            QLabel#capacityMix {
                color: #75d6a3; font-size: 14px; font-weight: 800;
                background: #1b302a; border-radius: 5px; padding: 8px;
            }
            QLabel#explodeValue {
                color: #f0a500; font-weight: 800; font-size: 15px;
            }
            QFrame#viewToolbar {
                background: #1b242d;
                border-bottom: 1px solid #35414d;
            }
            QPushButton#viewButton {
                background: #29343f; color: #cbd5df;
                border: 1px solid #465361; border-radius: 5px;
                padding: 6px 12px; font-size: 11px; font-weight: 800;
            }
            QPushButton#viewButton:hover {
                border-color: #f0a500; color: #ffffff;
            }
            QPushButton#viewButton:checked {
                background: #d99500; color: #171717;
                border-color: #f0a500;
            }
            QPushButton {
                background: #303b46; color: #edf2f6;
                border: 1px solid #465361; border-radius: 6px;
                padding: 9px 10px; font-weight: 700;
            }
            QPushButton:hover {
                background: #3a4855; border-color: #f0a500;
            }
            QPushButton:pressed { background: #202a33; }
            QPushButton:disabled {
                background: #222930; color: #68737e; border-color: #303840;
            }
            QPushButton#primaryButton {
                background: #d99500; color: #171717; border: 1px solid #f0a500;
                font-weight: 900;
            }
            QPushButton#primaryButton:hover { background: #ffb81c; }
            QDoubleSpinBox, QComboBox, QListWidget {
                background: #11171d; color: #edf2f6;
                border: 1px solid #3b4753; border-radius: 5px; padding: 6px;
                selection-background-color: #75530b;
            }
            QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {
                border: 1px solid #f0a500;
            }
            QCheckBox { spacing: 8px; color: #dce3e9; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border: 1px solid #667482;
                border-radius: 3px; background: #11171d;
            }
            QCheckBox::indicator:checked {
                background: #f0a500; border-color: #f0a500;
            }
            QSlider { accent-color: #f0a500; }
            QProgressBar {
                background: #11171d; border: 1px solid #35414d;
                border-radius: 4px; height: 8px;
            }
            QProgressBar::chunk { background: #f0a500; border-radius: 4px; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: #11171d; width: 11px; margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #53616e; min-height: 28px; border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover { background: #f0a500; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

    def _add_source_preview(self):
        """Render the original STL as a clearly visible, independently styled reference."""
        if self.generator.source is None or not self.show_source.isChecked():
            self.source_actor = None
            self.source_edge_actor = None
            return
        # Show the located plug after generation so the two female sockets
        # receiving the base locating pins are visible in the preview.
        # This is a display-only copy; STL export geometry is not changed.
        preview_mesh = (self.generator.plug.copy()
                        if self.generator.plug is not None
                        else self.generator.source.copy())
        if self.generator.parts:
            preview_mesh.apply_translation([
                0, self.generator.base_top + self.generator.lift_y, 0
            ])
        preview_mesh = transform_to_preview(preview_mesh)
        source_poly = pv.wrap(preview_mesh)
        self.source_actor = self.plotter.add_mesh(
            source_poly,
            color="#25d8cf",
            opacity=0.88,
            smooth_shading=True,
            ambient=0.28,
            diffuse=0.72,
            specular=0.55,
            specular_power=40,
            name="source",
        )
        self.source_edge_actor = None
        try:
            edges = source_poly.extract_feature_edges(
                feature_angle=8,
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False,
            )
            if edges.n_cells:
                self.source_edge_actor = self.plotter.add_mesh(
                    edges,
                    color="#073b3d",
                    line_width=2.0,
                    render_lines_as_tubes=True,
                    name="source_feature_edges",
                )
        except Exception:
            self.source_edge_actor = None

    def _add_part_preview(self, mesh, part, index):
        """Add a lit Z-up preview copy and feature-edge overlay."""
        color = "#68727e" if index == 0 else PALETTE[(index - 1) % len(PALETTE)]
        preview_mesh = transform_to_preview(mesh)
        surface = self.plotter.add_mesh(
            pv.wrap(preview_mesh),
            color=color,
            smooth_shading=True,
            show_edges=False,
            ambient=0.24,
            diffuse=0.78,
            specular=0.42,
            specular_power=32,
            name=part.name,
        )

        outline = None
        try:
            edge_mesh = pv.wrap(preview_mesh).extract_feature_edges(
                feature_angle=12,
                boundary_edges=True,
                non_manifold_edges=True,
                feature_edges=True,
                manifold_edges=False,
            )
            if edge_mesh.n_cells:
                outline = self.plotter.add_mesh(
                    edge_mesh,
                    color="#171b20",
                    line_width=2.0,
                    render_lines_as_tubes=True,
                    name=f"{part.name}_feature_edges",
                )
                outline.SetVisibility(True)
        except Exception:
            # The shaded solid remains usable if a PyVista version cannot
            # extract feature edges with these options.
            outline = None
        return surface, outline

    def _frame_preview(self):
        # Build a dimensioned XYZ grid from the currently displayed geometry.
        # Preview coordinates are Z-up: X = width, Y = length, Z = height.
        bounds_meshes = []
        if self.generator.parts:
            distance = (self.explode.value() / 100) * (
                self.wall.value() + max(12, max(self.generator.source.extents) * .12)
            )
            for i, part in enumerate(self.generator.parts):
                mesh = transform_to_preview(part.mesh)
                if i:
                    mesh.apply_translation(np.asarray(part.direction) * distance)
                bounds_meshes.append(mesh)
            if self.generator.source is not None and self.show_source.isChecked():
                source = (self.generator.plug.copy()
                          if self.generator.plug is not None
                          else self.generator.source.copy())
                source.apply_translation([
                    0, self.generator.base_top + self.generator.lift_y, 0
                ] if self.generator.parts else [0, 0, 0])
                bounds_meshes.append(transform_to_preview(source))
        elif self.generator.source is not None:
            bounds_meshes.append(transform_to_preview(self.generator.source))

        self.plotter.add_axes(line_width=2, color="#d7dde5")
        if bounds_meshes:
            low = np.min([m.bounds[0] for m in bounds_meshes], axis=0)
            high = np.max([m.bounds[1] for m in bounds_meshes], axis=0)
            size = np.maximum(high - low, 1.0)
            pad = np.maximum(size * .08, 2.0)
            bounds = (
                low[0] - pad[0], high[0] + pad[0],
                low[1] - pad[1], high[1] + pad[1],
                min(0.0, low[2] - pad[2]), high[2] + pad[2]
            )
            self.plotter.show_grid(
                bounds=bounds,
                xtitle="WIDTH (X) · mm",
                ytitle="LENGTH (Y) · mm",
                ztitle="HEIGHT (Z) · mm",
                n_xlabels=5,
                n_ylabels=5,
                n_zlabels=5,
                fmt="%.0f",
                color="#aebbc8",
                grid="back",
            )
            dimensions = (
                f"WIDTH  {size[0]:.1f} mm     "
                f"LENGTH  {size[1]:.1f} mm     "
                f"HEIGHT  {size[2]:.1f} mm"
            )
            self.plotter.add_text(
                dimensions, position="upper_left", font_size=10,
                color="#f0a500", name="model_dimensions"
            )
        else:
            self.plotter.show_grid(
                xtitle="WIDTH (X) · mm",
                ytitle="LENGTH (Y) · mm",
                ztitle="HEIGHT (Z) · mm",
                color="#aebbc8",
                grid="back",
            )
        self.plotter.reset_camera()
        try:
            # Use the same corrected ISO direction as the ISO toolbar button.
            # The negative Y component brings the model's front toward the viewer.
            self.set_visualiser_view("iso")
            self.plotter.camera.up = (0, 0, 1)
            self.plotter.camera.zoom(1.12)
        except Exception:
            pass
        self.plotter.render()

    def set_visualiser_view(self, view):
        """Switch the 3D viewport to a named orthographic-style camera angle."""
        directions = {
            "top": (0.0, 0.0, 1.0),
            # Keep Front aligned with the same -Y side used by the corrected ISO view.
            "front": (0.0, -1.0, 0.0),
            "back": (0.0, 1.0, 0.0),
            "left": (-1.0, 0.0, 0.0),
            "right": (1.0, 0.0, 0.0),
            # View from the front (-Y) side so the model faces the viewer.
            "iso": (1.0, -1.0, 0.85),
        }
        direction = np.asarray(directions.get(view, directions["iso"]), dtype=float)
        direction /= np.linalg.norm(direction)

        # Keep the current framing and zoom; only move the camera around its
        # focal point. The top view uses Y as its screen-up direction.
        try:
            camera = self.plotter.camera
            focal = np.asarray(camera.focal_point, dtype=float)
            current_position = np.asarray(camera.position, dtype=float)
            distance = float(np.linalg.norm(current_position - focal))
            if not np.isfinite(distance) or distance < 1e-6:
                self.plotter.reset_camera()
                camera = self.plotter.camera
                focal = np.asarray(camera.focal_point, dtype=float)
                distance = float(np.linalg.norm(
                    np.asarray(camera.position, dtype=float) - focal
                ))
            camera.position = tuple(focal + direction * distance)
            camera.focal_point = tuple(focal)
            camera.up = (0.0, 1.0, 0.0) if view == "top" else (0.0, 0.0, 1.0)
            self.plotter.render()
        except Exception:
            try:
                self.plotter.reset_camera()
                self.plotter.render()
            except Exception:
                pass

        for label, button in self.view_buttons.items():
            button.setChecked(label.lower() == view)

    def _project_settings(self):
        """Collect all user-adjustable mold settings plus fixed alignment parameters."""
        return {
            "mold_type": self.mold_style.currentData(),
            "panel_count": int(self.panel_count.currentData()),
            "top_extra": self.top_extra.value(),
            "wall": self.wall.value(),
            "base": self.base.value(),
            "clearance": self.clearance.value(),
            "fill_diameter": self.fill_dia.value(),
            "fill_depth": self.fill_depth.value(),
            "alignment": {
                "base_pin_radius": 1.9,
                "base_pin_height": 3.2,
                "panel_nub_radius": 1.5,
                "panel_socket_radius": 1.8,
                "panel_nub_depth": 2.8,
                "base_nub_radius": 1.0,
                "base_socket_radius": 1.25
            },
            "preview": {
                "show_source": self.show_source.isChecked(),
                "explode": self.explode.value()
            }
        }

    @staticmethod
    def _mesh_bytes(mesh):
        if mesh is None:
            return None
        return mesh.export(file_type="stl")

    @staticmethod
    def _mesh_from_bytes(data):
        mesh = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("A project mesh could not be restored.")
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        return mesh

    def save_project(self):
        if self.generator.source is None:
            QMessageBox.information(self, "Save project", "Load an STL before saving a project.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mold project", "Mad_Hat_Mold_Project.mhmold",
            "Mad Hat Mold Project (*.mhmold)")
        if not path:
            return
        if not path.lower().endswith(".mhmold"):
            path += ".mhmold"
        try:
            manifest = {
                "format": "MadHatMoldProject",
                "version": 1,
                "source_name": getattr(self, "source_filename", "source.stl"),
                "settings": self._project_settings(),
                "generator": {
                    "lift_y": self.generator.lift_y,
                    "base_top": self.generator.base_top,
                    "cavity_volume_mm3": self.generator.cavity_volume_mm3,
                    "pour_channel_volume_mm3": self.generator.pour_channel_volume_mm3,
                    "total_resin_volume_mm3": self.generator.total_resin_volume_mm3,
                    "parts": [
                        {"name": part.name, "direction": list(part.direction),
                         "mesh": f"geometry/part_{i:02d}.stl"}
                        for i, part in enumerate(self.generator.parts)
                    ],
                    "has_plug": self.generator.plug is not None
                }
            }
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("project.json", json.dumps(manifest, indent=2))
                archive.writestr("geometry/source_reference.stl",
                                 self._mesh_bytes(self.generator.source))
                if self.generator.plug is not None:
                    archive.writestr("geometry/located_plug.stl",
                                     self._mesh_bytes(self.generator.plug))
                for i, part in enumerate(self.generator.parts):
                    archive.writestr(f"geometry/part_{i:02d}.stl",
                                     self._mesh_bytes(part.mesh))
            self.status.setText(f"Project saved: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open mold project", "", "Mad Hat Mold Project (*.mhmold)")
        if not path:
            return
        try:
            with zipfile.ZipFile(path, "r") as archive:
                manifest = json.loads(archive.read("project.json").decode("utf-8"))
                if manifest.get("format") != "MadHatMoldProject":
                    raise ValueError("This is not a valid Mad Hat mold project.")
                if manifest.get("version") != 1:
                    raise ValueError("This project uses an unsupported version.")
                settings = manifest["settings"]
                gen_data = manifest["generator"]
                source = self._mesh_from_bytes(archive.read("geometry/source_reference.stl"))
                restored_parts = []
                for part_data in gen_data.get("parts", []):
                    mesh = self._mesh_from_bytes(archive.read(part_data["mesh"]))
                    restored_parts.append(MoldPart(
                        part_data["name"], mesh,
                        tuple(part_data.get("direction", (0.0, 0.0, 0.0)))))
                plug = None
                if gen_data.get("has_plug"):
                    plug = self._mesh_from_bytes(archive.read("geometry/located_plug.stl"))

            # Restore the original model and generated geometry without rerunning
            # booleans, so reopening a project reproduces the saved mould exactly.
            self.generator = MoldGenerator()
            self.generator.source = source
            self.source_original_mesh = source.copy()
            self._set_transform_controls_enabled(True)
            self.generator.parts = restored_parts
            self.generator.plug = plug
            self.generator.lift_y = float(gen_data.get("lift_y", 0.0))
            self.generator.base_top = float(gen_data.get("base_top", 0.0))
            self.generator.cavity_volume_mm3 = float(gen_data.get("cavity_volume_mm3", 0.0))
            self.generator.pour_channel_volume_mm3 = float(gen_data.get("pour_channel_volume_mm3", 0.0))
            self.generator.total_resin_volume_mm3 = float(gen_data.get("total_resin_volume_mm3", 0.0))

            self.mold_style.setCurrentIndex(max(0, self.mold_style.findData(settings.get("mold_type", "multi"))))
            self.panel_count.setCurrentIndex(max(0, self.panel_count.findData(int(settings.get("panel_count", 4)))))
            for key, widget in (("top_extra", self.top_extra), ("wall", self.wall),
                                ("base", self.base), ("clearance", self.clearance),
                                ("fill_diameter", self.fill_dia), ("fill_depth", self.fill_depth)):
                if key in settings:
                    widget.setValue(float(settings[key]))

            preview = settings.get("preview", {})
            self.show_source.setChecked(bool(preview.get("show_source", True)))
            self.plotter.clear()
            self.part_actors = []
            self.outline_actors = []
            self.visibility.clear()
            for i, part in enumerate(self.generator.parts):
                surface, outline = self._add_part_preview(part.mesh, part, i)
                self.part_actors.append(surface)
                self.outline_actors.append(outline)
                item = QListWidgetItem(part.name.replace("_", " "))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.visibility.addItem(item)
            self._add_source_preview()
            self._frame_preview()

            source_name = manifest.get("source_name", "source.stl")
            self.source_filename = source_name
            self.model_info.setText(
                f"{source_name}\\n{source.extents[0]:.1f} × "
                f"{source.extents[1]:.1f} × {source.extents[2]:.1f} mm")
            self.delete_stl_btn.setEnabled(True)
            self.save_project_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
            has_geometry = bool(self.generator.parts)
            for button in (self.export_selected, self.export_all,
                           self.export_exploded, self.export_plug):
                button.setEnabled(has_geometry)
            self.explode_btn.setEnabled(has_geometry)
            self.explode.setValue(int(preview.get("explode", 0)) if has_geometry else 0)
            self._update_capacity_display()
            self._update_stl_resin_display()
            self.status.setText(
                f"Project loaded: {os.path.basename(path)}. "
                "Saved mould geometry restored without regeneration.")
        except Exception as exc:
            QMessageBox.critical(self, "Load project failed", str(exc))

    def _update_stl_resin_display(self):
        """Show resin volume for the original STL solid only."""
        source = self.generator.source
        if source is None:
            self.stl_resin_volume.setText("Original STL volume: —")
            self.stl_resin_fill.setText("Resin to fill STL: —")
            self.stl_resin_mix.setText("Mixing amount (+10%): —")
            return

        volume_mm3 = abs(float(source.volume))
        volume_ml = volume_mm3 / 1000.0
        mixed_ml = volume_ml * 1.10
        self.stl_resin_volume.setText(
            f"Original STL volume: {volume_ml:,.1f} mL "
            f"({volume_ml / 1000.0:,.3f} L)")
        self.stl_resin_fill.setText(
            f"Resin to fill STL: {volume_ml:,.1f} mL "
            f"({volume_ml / 1000.0:,.3f} L)")
        self.stl_resin_mix.setText(
            f"Mixing amount (+10%): {mixed_ml:,.1f} mL "
            f"({mixed_ml / 1000.0:,.3f} L)")

    def _update_capacity_display(self):
        gap_ml = self.generator.cavity_volume_mm3 / 1000.0
        channel_ml = self.generator.pour_channel_volume_mm3 / 1000.0
        total_ml = self.generator.total_resin_volume_mm3 / 1000.0
        self.capacity_volume.setText(
            f"Silicone gap required: {gap_ml:,.1f} mL ({gap_ml/1000:,.3f} L)")
        self.capacity_channel.setText(
            f"Pour channel volume: {channel_ml:,.1f} mL ({channel_ml/1000:,.3f} L)")
        self.capacity_total.setText(
            f"Total silicone required: {total_ml:,.1f} mL ({total_ml/1000:,.3f} L)")
        self.capacity_extra.setText(
            f"Mixing amount (+10%): {total_ml*1.10:,.1f} mL "
            f"({total_ml*1.10/1000:,.3f} L)")

    def load_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add STL", "", "STL files (*.stl)")
        if not path:
            return
        try:
            mesh = trimesh.load(path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError("The selected STL does not contain a triangle mesh.")
            mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi/2, [1, 0, 0]))
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            mesh.apply_translation([-mesh.bounds.mean(axis=0)[0], -mesh.bounds[0, 1],
                                     -mesh.bounds.mean(axis=0)[2]])
            if not mesh.is_watertight:
                raise ValueError("The uploaded STL is not watertight. Repair it in a mesh tool and retry.")
            if self.stl_models:
                existing_max_x = max(float(m.bounds[1, 0]) for m in self.stl_models)
                mesh.apply_translation([
                    existing_max_x + max(5.0, float(mesh.extents[0]) * 0.10) - float(mesh.bounds[0, 0]),
                    0.0, 0.0
                ])
            self.stl_models.append(mesh.copy())
            self.stl_model_visible.append(True)
            name = os.path.basename(path)
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.stl_selection.addItem(item)
            self.stl_selection.setCurrentRow(self.stl_selection.count() - 1)
            self.source_filename = name
            self._rebuild_source_mesh()
            self._set_transform_controls_enabled(True)
            self.duplicate_btn.setEnabled(True)
            self.stl_show_btn.setEnabled(True)
            self.stl_hide_btn.setEnabled(True)
            self.save_project_btn.setEnabled(True)
            self.model_info.setText(f"{len(self.stl_models)} STL model(s) loaded · {sum(self.stl_model_visible)} visible")
            self._update_stl_resin_display()
            self.capacity_volume.setText("Silicone gap required: —")
            self.capacity_channel.setText("Pour channel volume: —")
            self.capacity_total.setText("Total silicone required: —")
            self.capacity_extra.setText("Mixing amount (+10%): —")
            self.delete_stl_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
            self.plotter.clear()
            self._add_source_preview()
            self._frame_preview()
            self.status.setText(f"Added {name}. Select an STL from the list to edit it.")
        except Exception as e:
            QMessageBox.critical(self, "STL load failed", str(e))

    def _rebuild_source_mesh(self):
        """Rebuild the mould source from visible STL entries only.

        An empty visible set is valid for preview purposes; generation is
        disabled until at least one model is shown again.
        """
        visible = [
            mesh.copy()
            for mesh, shown in zip(self.stl_models, self.stl_model_visible)
            if shown
        ]
        if not visible:
            self.generator.source = None
            self.generator.parts = []
            self.generator.plug = None
            self.source_original_mesh = None
            self.generate_btn.setEnabled(False)
            return

        combined = trimesh.util.concatenate(visible)
        combined.remove_unreferenced_vertices()
        combined.fix_normals()
        self.generator.source = combined
        self.source_original_mesh = combined.copy()
        self.generate_btn.setEnabled(True)

    def stl_selection_changed(self, row):
        enabled = 0 <= row < len(self.stl_models)
        for button in (self.stl_show_btn, self.stl_hide_btn,
                       self.stl_delete_selected_btn):
            button.setEnabled(enabled)
        self.stl_show_all_btn.setEnabled(bool(self.stl_models))
        if enabled:
            shown = self.stl_model_visible[row]
            self.status.setText(
                f"Selected: {self.stl_selection.item(row).text()} "
                f"({'visible' if shown else 'hidden'})"
            )

    def set_selected_stl_visibility(self, visible):
        row = self.stl_selection.currentRow()
        if not 0 <= row < len(self.stl_models):
            return

        visible = bool(visible)
        if self.stl_model_visible[row] == visible:
            return

        self.stl_model_visible[row] = visible
        item = self.stl_selection.item(row)
        item.setCheckState(Qt.Checked if visible else Qt.Unchecked)

        # Visibility is per STL entry, not the global "show original" preview
        # checkbox. Rebuild the source and redraw immediately.
        self._rebuild_source_mesh()
        self._refresh_source_scene()
        self._update_stl_resin_display()
        self.model_info.setText(
            f"{len(self.stl_models)} STL model(s) loaded · "
            f"{sum(self.stl_model_visible)} visible"
        )
        if self.generator.source is None:
            self.status.setText("All STL models are hidden. Show a model to generate a mould.")
        else:
            self.status.setText(
                f"{item.text()} {'shown' if visible else 'hidden'}."
            )

    def show_all_stls(self):
        """Make every imported STL visible again."""
        if not self.stl_models:
            return
        self.stl_model_visible = [True] * len(self.stl_models)
        for row in range(self.stl_selection.count()):
            self.stl_selection.item(row).setCheckState(Qt.Checked)
        self._rebuild_source_mesh()
        self._refresh_source_scene()
        self._update_stl_resin_display()
        self.model_info.setText(
            f"{len(self.stl_models)} STL model(s) loaded · "
            f"{len(self.stl_models)} visible"
        )
        self.status.setText("All STL models are now visible.")

    def delete_selected_stl(self):
        """Delete only the currently selected STL entry."""
        row = self.stl_selection.currentRow()
        if not 0 <= row < len(self.stl_models):
            return

        name = self.stl_selection.item(row).text()
        answer = QMessageBox.question(
            self, "Delete selected STL",
            f"Delete '{name}' from the project?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if answer != QMessageBox.Yes:
            return

        self.stl_models.pop(row)
        self.stl_model_visible.pop(row)
        self.stl_selection.takeItem(row)

        # Clear any mould generated from the previous source arrangement.
        self.generator.parts = []
        self.generator.plug = None
        self.generator.profile = None
        self._rebuild_source_mesh()
        self._refresh_source_scene()
        self._update_stl_resin_display()

        has_models = bool(self.stl_models)
        if has_models:
            new_row = min(row, len(self.stl_models) - 1)
            self.stl_selection.setCurrentRow(new_row)
            self.source_filename = self.stl_selection.item(new_row).text()
            self._set_transform_controls_enabled(True)
            self.duplicate_btn.setEnabled(True)
            self.save_project_btn.setEnabled(True)
            self.delete_stl_btn.setEnabled(True)
            self.model_info.setText(
                f"{len(self.stl_models)} STL model(s) loaded · "
                f"{sum(self.stl_model_visible)} visible"
            )
            self.status.setText(f"Deleted {name}.")
        else:
            self.source_filename = None
            self._set_transform_controls_enabled(False)
            self.duplicate_btn.setEnabled(False)
            self.save_project_btn.setEnabled(False)
            self.delete_stl_btn.setEnabled(False)
            self.model_info.setText("No model loaded.\\nUpload an STL to get started.")
            self.status.setText(f"Deleted {name}. No STL models remain.")
            self.capacity_volume.setText("Silicone gap required: —")
            self.capacity_channel.setText("Pour channel volume: —")
            self.capacity_total.setText("Total silicone required: —")
            self.capacity_extra.setText("Mixing amount (+10%): —")

        self.generate_btn.setEnabled(self.generator.source is not None)
        self.explode.setValue(0)
        self.explode_label.setText("0%")
        self.explode_btn.setEnabled(False)
        self.explode_btn.setText("EXPLODE MOLD")
        for button in (self.export_selected, self.export_all,
                       self.export_exploded, self.export_plug):
            button.setEnabled(False)

    def _refresh_source_scene(self):
        self.generator.parts = []
        self.generator.plug = None
        self.plotter.clear()
        self.part_actors = []
        self.outline_actors = []
        self.source_actor = None
        self.source_edge_actor = None
        self._add_source_preview()
        self._frame_preview()

    def delete_stl(self):
        """Remove the loaded STL and clear any mold generated from it."""
        if self.generator.source is None:
            return
        answer = QMessageBox.question(
            self, "Delete STL",
            "Delete the loaded STL and all generated mold parts?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return

        # Reset all model and generated geometry while preserving the UI settings.
        self.generator = MoldGenerator()
        self.source_original_mesh = None
        self.stl_models = []
        self.stl_model_visible = []
        self.stl_selection.clear()
        for button in (self.stl_show_btn, self.stl_hide_btn):
            button.setEnabled(False)
        self._set_transform_controls_enabled(False)
        self.duplicate_btn.setEnabled(False)
        self.plotter.clear()
        self.source_actor = None
        self.source_edge_actor = None
        self.part_actors = []
        self.outline_actors = []
        self.visibility.clear()
        self.model_info.setText("No model loaded.")
        self._update_stl_resin_display()
        self.capacity_volume.setText("Silicone gap required: —")
        self.capacity_channel.setText("Pour channel volume: —")
        self.capacity_total.setText("Total silicone required: —")
        self.capacity_extra.setText("Mixing amount (+10%): —")
        self.generate_btn.setEnabled(False)
        self.delete_stl_btn.setEnabled(False)
        self.save_project_btn.setEnabled(False)
        self.explode.setValue(0)
        self.explode_label.setText("0%")
        self.explode_btn.setText("EXPLODE MOLD")
        self.explode_btn.setEnabled(False)
        for button in (self.export_selected, self.export_all,
                       self.export_exploded, self.export_plug):
            button.setEnabled(False)
        self.status.setText("STL deleted. Upload a new STL to begin.")
        self._frame_preview()

    def _sync_uniform_scale(self, changed_axis, value):
        if not self.uniform_scale.isChecked():
            return
        targets = {"x": self.scale_x, "y": self.scale_y, "z": self.scale_z}
        for axis, widget in targets.items():
            if axis != changed_axis:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)

    def _set_transform_controls_enabled(self, enabled):
        for widget in (self.apply_transform_btn, self.reset_transform_btn):
            widget.setEnabled(bool(enabled))

    def apply_model_transform(self):
        """Apply transform controls to the currently selected STL only."""
        row = self.stl_selection.currentRow()
        if not 0 <= row < len(self.stl_models):
            return
        try:
            mesh = self.stl_models[row].copy()
            mesh.apply_scale([self.scale_x.value(), self.scale_y.value(), self.scale_z.value()])
            centre = mesh.bounds.mean(axis=0)
            for angle, axis in ((self.rotate_x.value(), [1, 0, 0]),
                                (self.rotate_y.value(), [0, 1, 0]),
                                (self.rotate_z.value(), [0, 0, 1])):
                if abs(angle) > 1e-9:
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(
                        math.radians(angle), axis, point=centre))
            mesh.apply_translation([
                -mesh.bounds.mean(axis=0)[0], -mesh.bounds[0, 1],
                -mesh.bounds.mean(axis=0)[2]])
            mesh.apply_translation([self.move_x.value(), self.move_y.value(), self.move_z.value()])
            mesh.fix_normals()
            if not mesh.is_watertight:
                raise ValueError("The transformed STL is no longer watertight.")
            self.stl_models[row] = mesh
            self._rebuild_source_mesh()
            self.generator.parts = []
            self.generator.plug = None
            self.generator.profile = None
            self.generator.lift_y = self.generator.base_top = 0.0
            self.generator.cavity_volume_mm3 = 0.0
            self.generator.pour_channel_volume_mm3 = 0.0
            self.generator.total_resin_volume_mm3 = 0.0
            self._refresh_source_scene()
            self._update_stl_resin_display()
            for label, value in ((self.capacity_volume, "Silicone gap required: —"),
                                 (self.capacity_channel, "Pour channel volume: —"),
                                 (self.capacity_total, "Total silicone required: —"),
                                 (self.capacity_extra, "Mixing amount (+10%): —")):
                label.setText(value)
            for button in (self.export_selected, self.export_all, self.export_exploded, self.export_plug):
                button.setEnabled(False)
            self.explode_btn.setEnabled(False)
            self.explode.setValue(0)
            self.status.setText("Selected STL transform applied. Regenerate the mould to update it.")
        except Exception as exc:
            QMessageBox.critical(self, "Transform failed", str(exc))

    def duplicate_model(self):
        """Duplicate only the currently selected STL model."""
        row = self.stl_selection.currentRow()
        if not 0 <= row < len(self.stl_models):
            return
        original = self.stl_models[row].copy()
        duplicate = original.copy()
        spacing = float(original.extents[0]) + max(5.0, float(original.extents[0]) * 0.10)
        duplicate.apply_translation([spacing, 0.0, 0.0])
        self.stl_models.append(duplicate)
        self.stl_model_visible.append(True)
        item = QListWidgetItem(f"{self.stl_selection.item(row).text()} (Copy)")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.stl_selection.addItem(item)
        self.stl_selection.setCurrentRow(self.stl_selection.count() - 1)
        self._rebuild_source_mesh()
        self._refresh_source_scene()
        self._update_stl_resin_display()
        self.status.setText("Selected STL duplicated as a separate model.")

    def reset_model_transform(self):
        if self.source_original_mesh is None:
            return
        for widget, value in (
            (self.scale_x, 1.0), (self.scale_y, 1.0), (self.scale_z, 1.0),
            (self.rotate_x, 0), (self.rotate_y, 0), (self.rotate_z, 0),
            (self.move_x, 0), (self.move_y, 0), (self.move_z, 0)
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.apply_model_transform()

    def _generation_progress(self, message):
        """Update the live progress readout while geometry is being built."""
        percent = 5
        if "Building fitted base" in message:
            percent = 30
        elif "Generating cavity panel" in message:
            try:
                current, total = message.split("panel", 1)[1].strip().split("/", 1)
                current = int(current)
                total = max(1, int(total.split("…", 1)[0].strip()))
                percent = 30 + int(55 * current / total)
            except (ValueError, IndexError):
                percent = 35
        elif "Building flat mold" in message:
            percent = 45
        self.generation_readout.setText(f"{percent}%  ·  {message}")
        self.status.setText(message)
        QApplication.processEvents()

    def generate(self):
        self.generate_btn.setEnabled(False)
        self.generation_readout.setText("5%  ·  Preparing mould generation…")
        self.progress.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.plotter.clear()
            if self.mold_style.currentData() == "flat":
                parts = self.generator.generate_flat_2d(
                    wall=self.wall.value(), base=self.base.value(),
                    top_extra=self.top_extra.value(), margin=10.0,
                    progress=self._generation_progress)
            else:
                parts=self.generator.generate(
                    panels=self.panel_count.currentData(),top_extra=self.top_extra.value(),
                    wall=self.wall.value(),base=self.base.value(),clearance=self.clearance.value(),
                    fill_diameter=self.fill_dia.value(),fill_depth=self.fill_depth.value(),
                    progress=self._generation_progress)
            self.part_actors=[]
            self.outline_actors=[]
            self.visibility.clear()
            for i,part in enumerate(parts):
                surface, outline = self._add_part_preview(part.mesh, part, i)
                self.part_actors.append(surface)
                self.outline_actors.append(outline)
                item=QListWidgetItem(part.name.replace("_"," "))
                item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked); self.visibility.addItem(item)
            self._add_source_preview()
            self._frame_preview()
            self.export_selected.setEnabled(True); self.export_all.setEnabled(True)
            self.export_exploded.setEnabled(True); self.export_plug.setEnabled(True)
            self.explode_btn.setEnabled(True)

            # Convert the actual gap volume (expanded offset solid minus
            # original STL solid) from mm³ to mL. Add a separate 10% mixing
            # allowance for practical pouring losses.
            gap_ml = self.generator.cavity_volume_mm3 / 1000.0
            channel_ml = self.generator.pour_channel_volume_mm3 / 1000.0
            total_ml = self.generator.total_resin_volume_mm3 / 1000.0
            recommended_ml = total_ml * 1.10

            gap_l = gap_ml / 1000.0
            channel_l = channel_ml / 1000.0
            total_l = total_ml / 1000.0
            recommended_l = recommended_ml / 1000.0

            self.capacity_volume.setText(
                f"Silicone gap required: {gap_ml:,.1f} mL  ({gap_l:,.3f} L)")
            self.capacity_channel.setText(
                f"Pour channel volume: {channel_ml:,.1f} mL  "
                f"({channel_l:,.3f} L)")
            self.capacity_total.setText(
                f"Total silicone required: {total_ml:,.1f} mL  "
                f"({total_l:,.3f} L)")
            self.capacity_extra.setText(
                f"Mixing amount (+10%): {recommended_ml:,.1f} mL  "
                f"({recommended_l:,.3f} L)")
            self.explode.setValue(0)
            self.explode_btn.setText("EXPLODE MOLD")
            if self.mold_style.currentData() == "flat":
                self.status.setText(
                    "Generated a one-piece 2D flat pour mold: raised positive plug, flat base "
                    "and surrounding silicone-retaining border.")
            else:
                self.status.setText(f"Generated 1 fitted base + {len(parts)-1} interlocking cavity panels. "
                                    f"Fill hole: {self.fill_dia.value():.1f} mm × {self.fill_depth.value():.1f} mm.")
            self.generation_readout.setText("100%  ·  Mould generation complete.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self,"Mold generation failed",str(e))
            self.status.setText("Generation failed. Try fewer panels or a simpler watertight STL.")
            self.generation_readout.setText("Generation failed. See the error message above.")
        finally:
            QApplication.restoreOverrideCursor(); self.progress.hide(); self.generate_btn.setEnabled(True)

    def toggle_exploded_view(self):
        """Toggle between the assembled mold and the fully exploded preview."""
        if not self.generator.parts:
            return
        if self.explode.value() > 0:
            self.explode.setValue(0)
            self.explode_btn.setText("EXPLODE MOLD")
        else:
            self.explode.setValue(100)
            self.explode_btn.setText("ASSEMBLE MOLD")

    def apply_explode(self,value):
        self.explode_value=value; self.explode_label.setText(f"{value}%")
        if hasattr(self, "explode_btn"):
            self.explode_btn.setText(
                "ASSEMBLE MOLD" if value > 0 else "EXPLODE MOLD"
            )
        if not self.generator.parts: return
        # Rebuild preview actors at their exploded positions.
        for actor in self.part_actors + [a for a in self.outline_actors if a is not None]:
            try: self.plotter.remove_actor(actor,render=False)
            except Exception: pass
        self.part_actors=[]
        self.outline_actors=[]
        distance=(value/100)*(self.wall.value()+max(12,max(self.generator.source.extents)*.12))
        for i,part in enumerate(self.generator.parts):
            # Preserve each part's checkbox state when rebuilding actors for
            # the new exploded position. Re-creating an actor defaults it to
            # visible, so explicitly restore the user's visibility choice.
            item = self.visibility.item(i) if i < self.visibility.count() else None
            part_visible = (
                item is None or item.checkState() == Qt.Checked
            )

            mesh=part.mesh.copy()
            if i: mesh.apply_translation(np.array(part.direction)*distance)
            surface, outline = self._add_part_preview(mesh, part, i)

            surface.SetVisibility(part_visible)
            if outline is not None:
                outline.SetVisibility(part_visible)

            self.part_actors.append(surface)
            self.outline_actors.append(outline)
        self.plotter.render()

    def visibility_changed(self,item):
        idx=self.visibility.row(item)
        if 0<=idx<len(self.part_actors):
            visible = item.checkState()==Qt.Checked
            self.part_actors[idx].SetVisibility(visible)
            if idx < len(self.outline_actors) and self.outline_actors[idx] is not None:
                self.outline_actors[idx].SetVisibility(visible)
            self.plotter.render()

    def toggle_source(self,checked):
        # Source surface and its edge overlay are separate from mold actors/outlines.
        for name in ("source", "source_feature_edges"):
            try:
                self.plotter.remove_actor(name, render=False)
            except Exception:
                pass
        self.source_actor = None
        self.source_edge_actor = None
        if checked:
            self._add_source_preview()
        self.plotter.render()

    def export_one(self):
        idx=self.visibility.currentRow()
        if idx<0: idx=0
        if idx>=len(self.generator.parts): return
        name=self.generator.parts[idx].name+".stl"
        path,_=QFileDialog.getSaveFileName(self,"Export STL",name,"STL files (*.stl)")
        if path:
            try: self.generator.export_part(idx,path); self.status.setText(f"Exported {path}")
            except Exception as e: QMessageBox.critical(self,"Export failed",str(e))

    def export_all_parts(self):
        path,_=QFileDialog.getSaveFileName(self,"Export all parts","Mad_Hat_Mold_Parts.zip","ZIP files (*.zip)")
        if path:
            try: self.generator.export_zip(path,0); self.status.setText(f"Exported {path}")
            except Exception as e: QMessageBox.critical(self,"Export failed",str(e))

    def export_exploded_parts(self):
        path,_=QFileDialog.getSaveFileName(self,"Export exploded mold","Mad_Hat_Mold_Exploded.zip","ZIP files (*.zip)")
        if path:
            try: self.generator.export_zip(path,self.explode_value); self.status.setText(f"Exported {path}")
            except Exception as e: QMessageBox.critical(self,"Export failed",str(e))

    def export_plug_stl(self):
        if self.generator.plug is None: return
        path,_=QFileDialog.getSaveFileName(self,"Export located original plug",
            "Mad_Hat_Mold_Plug_With_Locating_Holes.stl","STL files (*.stl)")
        if path:
            try: transform_to_export(self.generator.plug).export(path,file_type="stl")
            except Exception as e: QMessageBox.critical(self,"Export failed",str(e))


def main():
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
