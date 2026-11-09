"""
MAD HAT — Negative Mold Generator (Python desktop edition)

Install:
    py -m pip install PySide6 pyvista pyvistaqt trimesh manifold3d numpy scipy

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
import traceback
from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QFileDialog, QDoubleSpinBox, QSpinBox,
    QComboBox, QSlider, QCheckBox, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QSplitter, QScrollArea, QFrame
)
try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "Missing 3D preview packages. Install with:\n"
        "py -m pip install PySide6 pyvista pyvistaqt trimesh manifold3d numpy scipy"
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


def radial_profile(mesh, height, angle_count=192, y_count=96):
    """
    Build a stable radial envelope from the source mesh.

    Vertex bins can be sparse and can leave isolated spikes/gaps.  Fill each
    angular row periodically, interpolate empty height rows, then apply a
    restrained 2-D Gaussian smooth.  This removes sampling noise without
    introducing the large neighbourhood-max bulges that caused cavity artefacts.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    radii = np.full((y_count + 1, angle_count), np.nan, dtype=float)

    yi = np.clip(np.rint(v[:, 1] / max(height, 1e-8) * y_count).astype(int),
                 0, y_count)
    angles = (np.arctan2(v[:, 0], v[:, 2]) + 2 * np.pi) % (2 * np.pi)
    ai = np.floor(angles / (2 * np.pi) * angle_count).astype(int) % angle_count
    rr = np.hypot(v[:, 0], v[:, 2])

    # Keep the furthest observed vertex in each bin; this is the conservative
    # radial silhouette needed for a negative cavity.
    for row, col, radius in zip(yi, ai, rr):
        old = radii[row, col]
        if not np.isfinite(old) or radius > old:
            radii[row, col] = radius

    angles_grid = np.arange(angle_count + 1, dtype=float)
    for row in range(y_count + 1):
        valid = np.isfinite(radii[row])
        if valid.sum() >= 2:
            xp = np.flatnonzero(valid).astype(float)
            fp = radii[row, valid]
            xp_ext = np.r_[xp[-1] - angle_count, xp, xp[0] + angle_count]
            fp_ext = np.r_[fp[-1], fp, fp[0]]
            radii[row] = np.interp(angles_grid[:-1], xp_ext, fp_ext)
        elif valid.sum() == 1:
            radii[row] = radii[row, valid][0]

    # Interpolate rows that contain no source vertices.
    row_has_data = np.isfinite(radii).any(axis=1)
    if not row_has_data.any():
        raise ValueError("Could not build a radial profile from this STL.")
    known_rows = np.flatnonzero(row_has_data)
    for col in range(angle_count):
        known = np.isfinite(radii[:, col])
        if known.sum() >= 2:
            radii[:, col] = np.interp(np.arange(y_count + 1),
                                      np.flatnonzero(known), radii[known, col])
        elif known.sum() == 1:
            radii[:, col] = radii[known, col][0]
        else:
            radii[:, col] = np.interp(np.arange(y_count + 1), known_rows,
                                      np.nanmean(radii[known_rows], axis=1))

    fallback = max(mesh.extents[0], mesh.extents[2]) / 2
    radii = np.nan_to_num(radii, nan=fallback, posinf=fallback, neginf=fallback)

    # Smooth angular and vertical sampling noise.  A small outward safety pad
    # keeps the generated cavity from cutting into the source model.
    smooth = gaussian_filter(radii, sigma=(0.8, 1.15), mode=("nearest", "wrap"))
    smooth = np.maximum(smooth, radii - max(0.15, fallback * 0.001))
    pad = max(0.35, min(1.0, max(mesh.extents[0], mesh.extents[2]) * 0.002))
    smooth += pad

    # Smooth the external wall separately so the wall remains visually clean.
    outer = gaussian_filter(smooth, sigma=(1.2, 2.0), mode=("nearest", "wrap"))
    first = int(math.ceil(0.90 * y_count))
    opening = float(np.max(smooth[first]))
    smooth[first:] = opening
    outer[first:] = max(opening, float(np.max(outer[first:])))
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


def radial_cavity_cutter(profile, clearance, y_extra=0.75, count=192):
    """Create a clean watertight lofted cutter from the smoothed radial profile."""
    height = float(profile["height"])
    y0, y1 = -float(y_extra), height + float(y_extra)
    rows = max(64, int(profile["rows"]) * 2)
    verts, faces = [], []

    for j in range(rows + 1):
        y = y0 + (y1 - y0) * j / rows
        source_y = float(np.clip(y, 0.0, height))
        for i in range(count):
            angle = 2.0 * math.pi * i / count
            radius = max(0.2, profile_radius(profile, angle, source_y)
                         + max(0.0, float(clearance)))
            verts.append([math.sin(angle) * radius, y,
                          math.cos(angle) * radius])

    for j in range(rows):
        for i in range(count):
            ni = (i + 1) % count
            a, b = j * count + i, j * count + ni
            c, d = (j + 1) * count + ni, (j + 1) * count + i
            faces.extend([[a, b, d], [b, c, d]])

    bottom_center = len(verts)
    verts.append([0.0, y0, 0.0])
    top_center = len(verts)
    verts.append([0.0, y1, 0.0])
    top_start = rows * count
    for i in range(count):
        ni = (i + 1) % count
        faces.append([bottom_center, ni, i])
        faces.append([top_center, top_start + i, top_start + ni])

    cutter = trimesh.Trimesh(vertices=np.asarray(verts),
                             faces=np.asarray(faces), process=True)
    cutter.remove_unreferenced_vertices()
    cutter.fix_normals()
    return cutter


def offset_model(mesh, clearance):
    """Approximate normal offset using trimesh's vertex normals."""
    out = mesh.copy()
    normals = out.vertex_normals
    out.vertices += normals * (max(0.0, clearance) + .15)
    out.fix_normals()
    return out


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
        # The source model is normalized so its lowest point is Y=0.
        # End the base exactly at that plane: this gives the STL a true
        # face-to-face seat instead of burying its bottom inside the base.
        actual_base=max(1,base-1.0)
        base_top=0.0
        y_max=base_top+size[1]+max(0,float(top_extra))
        profile=radial_profile(src,size[1])
        self.profile=profile
        # A continuous radial loft avoids transferring source triangle noise
        # into the cavity, which caused random Boolean surface artefacts.
        cutter = radial_cavity_cutter(profile, clearance)
        cutter.apply_translation([0, base_top, 0])
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
            # Panel-to-panel alignment pins at top and bottom of each seam.
            pin_r=min(2.5,max(1.5,wall*.22))
            male_r=max(.8,pin_r-.2)
            inset=max(pin_r+.8,min(wall*.55,wall-1))
            seam=a1
            for yy in (y_min+pin_r+.5,y_max-pin_r-.5):
                rr=profile_radius(profile,seam,yy)+inset
                point=[math.sin(seam)*rr,yy,math.cos(seam)*rr]
                tangent=[math.cos(seam),0,-math.sin(seam)]
                male=cylinder_between(male_r*1.0,2.0,point-np.array(tangent),tangent)
                panel=boolean([panel,male],"union")
            # Contoured external clip flanges. Their full height follows the
            # smoothed outer wall profile instead of using a rectangular block.
            # Opposing tabs are shifted tangentially by half their thickness,
            # matching the HTML generator's edge-to-edge arrangement.
            flange_projection = 12.0
            flange_thickness = 3.0
            flange_offset = flange_thickness / 2
            for edge, angle in ((0, a0), (1, a1)):
                tangential_offset = flange_offset if edge == 0 else -flange_offset
                flange = contoured_clip_flange(
                    profile, angle, y_min, y_max, base_top,
                    clearance=clearance, wall=wall,
                    projection=flange_projection,
                    thickness=flange_thickness,
                    tangential_offset=tangential_offset,
                )
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
        self.explode_value=0
        self.source_actor=None
        self.source_edge_actor=None
        self.part_actors=[]
        self.outline_actors=[]
        self._build_ui()
        self._style()

    def _build_ui(self):
        root=QWidget(); layout=QHBoxLayout(root)
        split=QSplitter(Qt.Horizontal)
        left=QWidget(); side=QVBoxLayout(left)
        side.setContentsMargins(12,12,12,12)
        title=QLabel("MAD HAT · NEGATIVE MOLD GENERATOR")
        title.setStyleSheet(f"font-size:19px;font-weight:bold;color:{GOLD}")
        side.addWidget(title)
        subtitle=QLabel("Multi-piece negative cavity mold generator · Python desktop")
        subtitle.setStyleSheet("color:#aaa")
        side.addWidget(subtitle)

        source=QGroupBox("1 · Source model"); sf=QVBoxLayout(source)
        self.load_btn=QPushButton("UPLOAD STL MODEL")
        self.load_btn.clicked.connect(self.load_stl)
        sf.addWidget(self.load_btn)
        self.model_info=QLabel("No model loaded."); self.model_info.setWordWrap(True)
        sf.addWidget(self.model_info); side.addWidget(source)

        opts=QGroupBox("2 · Mold layout"); form=QFormLayout(opts)
        self.panel_count=QComboBox()
        for n in (4,5,6,8): self.panel_count.addItem(f"{n} panels",n)
        self.top_extra=self.spin(8,0,100,1)
        self.wall=self.spin(8,3,100,1)
        self.base=self.spin(6,3,100,1)
        self.clearance=self.spin(3,0,20,.1)
        self.fill_dia=self.spin(12,2,50,.5)
        self.fill_depth=self.spin(20,1,500,1)
        form.addRow("Removable side panels",self.panel_count)
        form.addRow("Panel height extra (mm)",self.top_extra)
        form.addRow("Wall thickness (mm)",self.wall)
        form.addRow("Base thickness (mm)",self.base)
        form.addRow("Cavity clearance (mm)",self.clearance)
        form.addRow("Silicone fill hole diameter (mm)",self.fill_dia)
        form.addRow("Fill hole depth (mm)",self.fill_depth)
        side.addWidget(opts)

        self.generate_btn=QPushButton("GENERATE MOLD")
        self.generate_btn.setEnabled(False); self.generate_btn.clicked.connect(self.generate)
        side.addWidget(self.generate_btn)
        self.status=QLabel("Upload an STL to begin."); self.status.setWordWrap(True)
        side.addWidget(self.status)

        exp=QGroupBox("3 · Exploded view / export"); ef=QVBoxLayout(exp)
        self.explode=QSlider(Qt.Horizontal); self.explode.setRange(0,100)
        self.explode.valueChanged.connect(self.apply_explode)
        ef.addWidget(QLabel("Explode amount")); ef.addWidget(self.explode)
        self.explode_label=QLabel("0%"); ef.addWidget(self.explode_label)
        row=QHBoxLayout()
        self.show_source=QCheckBox("Show source model"); self.show_source.setChecked(True)
        self.show_source.toggled.connect(self.toggle_source)
        self.show_source_edges=QCheckBox("Show STL edges"); self.show_source_edges.setChecked(True)
        self.show_source_edges.toggled.connect(self.toggle_source_edges)
        row.addWidget(self.show_source)
        row.addWidget(self.show_source_edges)
        ef.addLayout(row)
        self.show_outlines=QCheckBox("Show wall outlines")
        self.show_outlines.setChecked(True)
        self.show_outlines.toggled.connect(self.toggle_outlines)
        ef.addWidget(self.show_outlines)
        preview_hint=QLabel("Shaded surfaces, dark feature edges and an isometric view make the wall shape easier to read.")
        preview_hint.setWordWrap(True)
        preview_hint.setStyleSheet("color:#999;font-size:11px;")
        ef.addWidget(preview_hint)
        self.visibility=QListWidget(); self.visibility.itemChanged.connect(self.visibility_changed)
        ef.addWidget(self.visibility)
        self.export_selected=QPushButton("EXPORT SELECTED PART STL")
        self.export_selected.clicked.connect(self.export_one); ef.addWidget(self.export_selected)
        self.export_all=QPushButton("EXPORT ALL PARTS (ZIP)")
        self.export_all.clicked.connect(self.export_all_parts); ef.addWidget(self.export_all)
        self.export_exploded=QPushButton("EXPORT EXPLODED MOLD (ZIP)")
        self.export_exploded.clicked.connect(self.export_exploded_parts); ef.addWidget(self.export_exploded)
        self.export_plug=QPushButton("EXPORT LOCATED ORIGINAL PLUG STL")
        self.export_plug.clicked.connect(self.export_plug_stl); ef.addWidget(self.export_plug)
        for b in (self.export_selected,self.export_all,self.export_exploded,self.export_plug):
            b.setEnabled(False)
        side.addWidget(exp,1)

        self.progress=QProgressBar(); self.progress.setRange(0,0); self.progress.hide()
        side.addWidget(self.progress)
        side.addStretch(1)

        self.plotter=QtInteractor(self)
        self.plotter.set_background("#11151b", top="#28313d")
        self.plotter.add_axes(line_width=2, color="#d7dde5")
        self.plotter.show_grid(color="#39424e")
        try:
            self.plotter.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        # Put the complete settings/control panel inside a vertical scroll area so
        # every setting remains accessible on shorter screens.
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        settings_scroll.setWidget(left)

        split.addWidget(settings_scroll); split.addWidget(self.plotter.interactor)
        split.setSizes([370,1100])
        layout.addWidget(split)
        self.setCentralWidget(root)

    @staticmethod
    def spin(value, minimum, maximum, step):
        w=QDoubleSpinBox(); w.setRange(minimum,maximum); w.setSingleStep(step); w.setValue(value)
        w.setDecimals(2 if step<1 else 0)
        return w

    def _style(self):
        self.setStyleSheet("""
            QMainWindow,QWidget { background:#151515; color:#eee; font:13px Arial; }
            QGroupBox { border:1px solid #444; border-radius:7px; margin-top:10px;
                        padding:9px; font-weight:bold; color:#f0a500; }
            QGroupBox::title { subcontrol-origin:margin; left:9px; padding:0 4px; }
            QPushButton { background:#353535; color:#fff; border:1px solid #555;
                          border-radius:6px; padding:8px; font-weight:bold; }
            QPushButton:hover { border-color:#f0a500; }
            QPushButton:disabled { color:#777; }
            QDoubleSpinBox,QComboBox,QListWidget { background:#101010; color:#fff;
                border:1px solid #444; border-radius:5px; padding:5px; }
            QSlider { accent-color:#f0a500; }
            QScrollArea { border: none; }
            QScrollBar:vertical { background:#101010; width:12px; margin:2px; border-radius:5px; }
            QScrollBar::handle:vertical { background:#666; min-height:28px; border-radius:5px; }
            QScrollBar::handle:vertical:hover { background:#f0a500; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
        """)

    def _add_source_preview(self):
        """Render the original STL as a clearly visible, independently styled reference."""
        if self.generator.source is None or not self.show_source.isChecked():
            self.source_actor = None
            self.source_edge_actor = None
            return
        # Mold parts are lifted together to put the base on the build plate.
        # Apply that same display-only lift to the source STL so it remains
        # seated on the base instead of appearing to pass through it.
        source_preview = self.generator.source.copy()
        source_preview.apply_translation([0.0, self.generator.lift_y, 0.0])
        source_poly = pv.wrap(source_preview)
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
        if self.show_source_edges.isChecked():
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
        """Add a lit solid and a separate feature-edge overlay for clear part boundaries."""
        color = "#68727e" if index == 0 else PALETTE[(index - 1) % len(PALETTE)]
        surface = self.plotter.add_mesh(
            pv.wrap(mesh),
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
            edge_mesh = pv.wrap(mesh).extract_feature_edges(
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
                outline.SetVisibility(self.show_outlines.isChecked())
        except Exception:
            # The shaded solid remains usable if a PyVista version cannot
            # extract feature edges with these options.
            outline = None
        return surface, outline

    def _frame_preview(self):
        self.plotter.add_axes(line_width=2, color="#d7dde5")
        self.plotter.reset_camera()
        try:
            self.plotter.view_isometric()
            self.plotter.camera.zoom(1.12)
        except Exception:
            pass
        self.plotter.render()

    def load_stl(self):
        path,_=QFileDialog.getOpenFileName(self,"Open STL","","STL files (*.stl)")
        if not path: return
        try:
            size=self.generator.load_stl(path)
            self.model_info.setText(f"{os.path.basename(path)}\\n{size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm")
            self.generate_btn.setEnabled(True)
            self.plotter.clear()
            self._add_source_preview()
            self._frame_preview()
            self.status.setText("STL loaded. Set options, then generate.")
        except Exception as e:
            QMessageBox.critical(self,"STL load failed",str(e))

    def generate(self):
        self.generate_btn.setEnabled(False); self.progress.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.plotter.clear()
            parts=self.generator.generate(
                panels=self.panel_count.currentData(),top_extra=self.top_extra.value(),
                wall=self.wall.value(),base=self.base.value(),clearance=self.clearance.value(),
                fill_diameter=self.fill_dia.value(),fill_depth=self.fill_depth.value(),
                progress=lambda msg:self.status.setText(msg))
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
            self.status.setText(f"Generated 1 fitted base + {len(parts)-1} interlocking cavity panels. "
                                f"Fill hole: {self.fill_dia.value():.1f} mm × {self.fill_depth.value():.1f} mm.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self,"Mold generation failed",str(e))
            self.status.setText("Generation failed. Try fewer panels or a simpler watertight STL.")
        finally:
            QApplication.restoreOverrideCursor(); self.progress.hide(); self.generate_btn.setEnabled(True)

    def apply_explode(self,value):
        self.explode_value=value; self.explode_label.setText(f"{value}%")
        if not self.generator.parts: return
        # Rebuild preview actors at their exploded positions.
        for actor in self.part_actors + [a for a in self.outline_actors if a is not None]:
            try: self.plotter.remove_actor(actor,render=False)
            except Exception: pass
        self.part_actors=[]
        self.outline_actors=[]
        distance=(value/100)*(self.wall.value()+max(12,max(self.generator.source.extents)*.12))
        for i,part in enumerate(self.generator.parts):
            mesh=part.mesh.copy()
            if i: mesh.apply_translation(np.array(part.direction)*distance)
            surface, outline = self._add_part_preview(mesh, part, i)
            self.part_actors.append(surface)
            self.outline_actors.append(outline)
        self.plotter.render()

    def visibility_changed(self,item):
        idx=self.visibility.row(item)
        if 0<=idx<len(self.part_actors):
            visible = item.checkState()==Qt.Checked
            self.part_actors[idx].SetVisibility(visible)
            if idx < len(self.outline_actors) and self.outline_actors[idx] is not None:
                self.outline_actors[idx].SetVisibility(visible and self.show_outlines.isChecked())
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

    def toggle_source_edges(self,checked):
        try:
            self.plotter.remove_actor("source_feature_edges", render=False)
        except Exception:
            pass
        self.source_edge_actor = None
        if checked and self.show_source.isChecked() and self.generator.source is not None:
            try:
                edges = pv.wrap(self.generator.source).extract_feature_edges(
                    feature_angle=8,
                    boundary_edges=True,
                    non_manifold_edges=True,
                    feature_edges=True,
                    manifold_edges=False,
                )
                if edges.n_cells:
                    self.source_edge_actor = self.plotter.add_mesh(
                        edges, color="#073b3d", line_width=2.0,
                        render_lines_as_tubes=True, name="source_feature_edges")
            except Exception:
                self.source_edge_actor = None
        self.plotter.render()

    def toggle_outlines(self,checked):
        # Outlines are independent actors, so they can be switched off without
        # changing the STL geometry or the exported meshes.
        for i,actor in enumerate(self.outline_actors):
            if actor is not None:
                part_visible = (i >= self.visibility.count() or
                                self.visibility.item(i).checkState()==Qt.Checked)
                actor.SetVisibility(bool(checked and part_visible))
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
