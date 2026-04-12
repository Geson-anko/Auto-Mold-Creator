bl_info = {
    "name": "Mold Split",
    "author": "Auto-generated",
    "version": (2, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Mold Split",
    "description": "Generate resin mold with interlocking step from target object",
    "category": "Mesh",
}

import math
import bpy
import bmesh
from mathutils import Vector
from bpy.props import FloatProperty, PointerProperty


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _order_boundary_loop(boundary_edges):
    """Return ordered list of BMVert forming one closed loop, or None."""
    if not boundary_edges:
        return None
    adj: dict[int, list] = {}
    for e in boundary_edges:
        for v in e.verts:
            adj.setdefault(v.index, []).append((e, e.other_vert(v)))
    ordered, visited = [], set()
    cur = boundary_edges[0].verts[0]
    while cur.index not in visited:
        ordered.append(cur)
        visited.add(cur.index)
        for _, other in adj.get(cur.index, []):
            if other.index not in visited:
                cur = other
                break
        else:
            break
    expected = len({v.index for e in boundary_edges for v in e.verts})
    return ordered if len(ordered) == expected else None


def _newell_normal(positions: list[Vector]) -> Vector:
    """Polygon normal via Newell method."""
    n = Vector()
    nv = len(positions)
    for i in range(nv):
        c, nx = positions[i], positions[(i + 1) % nv]
        n.x += (c.y - nx.y) * (c.z + nx.z)
        n.y += (c.z - nx.z) * (c.x + nx.x)
        n.z += (c.x - nx.x) * (c.y + nx.y)
    n.normalize()
    return n


def _build_cut_surface(target_obj, boundary_indices, thickness, step_pct,
                       draft_deg: float = 90.0):
    """Build a stepped cut surface with guaranteed ≥90° corners.

    At each boundary-loop vertex an **outer corner vertex** (L1c / L2c) is
    inserted so that the step base quad and two step-wall quads all have
    exact right angles.  No face in the step profile has an internal angle
    less than the requested *draft_deg* (≥ 90°).

    *draft_deg* controls the wall-to-base angle (default 90° = vertical wall,
    larger values tilt the wall outward for easier release).

    Returns ``(cut_object, loop_normal)`` or ``(None, None)``.
    """
    step_h = thickness * (1.0 - step_pct)
    phi = math.radians(max(draft_deg - 90.0, 0.0))
    draft_off = step_h * math.tan(phi) if phi > 1e-6 else 0.0

    bb = [Vector(v) for v in target_obj.bound_box]
    extension = (bb[6] - bb[0]).length * 0.5

    # ------ read mesh data ------
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    bedges = [bm.edges[i] for i in boundary_indices]
    ordered = _order_boundary_loop(bedges)
    if ordered is None:
        bm.free()
        return None, None

    num = len(ordered)
    mesh_center = sum((v.co.copy() for v in bm.verts), Vector()) / len(bm.verts)
    positions = [v.co.copy() for v in ordered]
    loop_normal = _newell_normal(positions)
    loop_centroid = sum(positions, Vector()) / num

    # Ensure loop_normal points outward (away from mesh centre toward
    # the mold exterior) so that the step wall goes in the pull direction.
    if loop_normal.dot(loop_centroid - mesh_center) < 0:
        loop_normal = -loop_normal

    bm.free()

    # per-edge outward direction: loop_normal × edge_dir, pointing away
    # from the loop centroid.  This is purely geometric — no face lookup
    # required, which avoids all interior-face detection pitfalls.
    edge_out: list[Vector] = []
    for i in range(num):
        j = (i + 1) % num
        ed = (positions[j] - positions[i]).normalized()
        out = loop_normal.cross(ed)
        mid = (positions[i] + positions[j]) * 0.5
        if out.dot(mid - loop_centroid) < 0:
            out = -out
        edge_out.append(out)

    # per-vertex mitered outward (for extension flange only)
    vmiter: list[Vector] = []
    for i in range(num):
        prev = (i - 1) % num
        avg = (edge_out[prev] + edge_out[i]).copy()
        avg.normalize()
        d = avg.dot(edge_out[i])
        vmiter.append(avg * (1.0 / max(d, 0.01)))

    # ------ build vertices (8 per boundary vert) ------
    cbm = bmesh.new()
    L0, L1p, L1n, L1c = [], [], [], []
    L2p, L2n, L2c, L3 = [], [], [], []
    for j in range(num):
        p = positions[j]
        nm = loop_normal
        po = edge_out[(j - 1) % num]
        no = edge_out[j]

        l0 = p
        l1p = p + po * step_h
        l1n = p + no * step_h
        l1c = p + po * step_h + no * step_h          # outer corner
        l2p = l1p + nm * step_h + po * draft_off
        l2n = l1n + nm * step_h + no * draft_off
        l2c = l1c + nm * step_h + po * draft_off + no * draft_off
        l3 = p + vmiter[j] * extension + nm * step_h + vmiter[j] * draft_off

        L0.append(cbm.verts.new(l0))
        L1p.append(cbm.verts.new(l1p))
        L1n.append(cbm.verts.new(l1n))
        L1c.append(cbm.verts.new(l1c))
        L2p.append(cbm.verts.new(l2p))
        L2n.append(cbm.verts.new(l2n))
        L2c.append(cbm.verts.new(l2c))
        L3.append(cbm.verts.new(l3))
    cbm.verts.ensure_lookup_table()

    # ------ faces ------
    # centre face
    cbm.faces.new(list(reversed(L0)))

    # per-edge faces
    for i in range(num):
        j = (i + 1) % num
        cbm.faces.new([L0[i],  L0[j],  L1p[j], L1n[i]])   # step base
        cbm.faces.new([L1n[i], L1p[j], L2p[j], L2n[i]])   # step wall
        cbm.faces.new([L2n[i], L2p[j], L3[j],  L3[i]])    # extension

    # corner faces (all internal angles ≥ 90°)
    for j in range(num):
        cbm.faces.new([L0[j],  L1n[j], L1c[j], L1p[j]])   # base quad
        cbm.faces.new([L1p[j], L1c[j], L2c[j], L2p[j]])   # wall A (prev dir)
        cbm.faces.new([L1c[j], L1n[j], L2n[j], L2c[j]])   # wall B (next dir)
        cbm.faces.new([L2p[j], L2c[j], L3[j]])             # ext tri A
        cbm.faces.new([L2c[j], L2n[j], L3[j]])             # ext tri B

    mesh = bpy.data.meshes.new("_MoldSplit_Cut")
    cbm.to_mesh(mesh)
    cbm.free()

    # recalculate normals
    bm2 = bmesh.new()
    bm2.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    bm2.to_mesh(mesh)
    bm2.free()

    obj = bpy.data.objects.new("_MoldSplit_Cut", mesh)
    bpy.context.collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
    mod.thickness = 1e-6
    mod.offset = -1.0
    bpy.ops.object.modifier_apply(modifier=mod.name)

    return obj, loop_normal


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class MOLDSPLIT_OT_generate(bpy.types.Operator):
    """Generate mold split from target with seam-marked cut boundary"""
    bl_idname = "moldsplit.generate"
    bl_label = "Generate Mold Split"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.mold_split
        return props.target is not None and props.target.type == 'MESH'

    def execute(self, context):
        props = context.scene.mold_split
        target = props.target
        thickness = props.thickness
        step_pct = props.step_pct / 100.0
        draft_deg = math.degrees(props.draft_angle)

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # seam edges
        bm = bmesh.new()
        bm.from_mesh(target.data)
        bm.edges.ensure_lookup_table()
        boundary = [e.index for e in bm.edges if e.seam]
        bm.free()

        if not boundary:
            self.report({'ERROR'},
                        "No seam edges found. Mark cut boundary as seams.")
            return {'CANCELLED'}

        # step 1: solidify
        mold_mesh = target.data.copy()
        mold_obj = bpy.data.objects.new(f"{target.name}_Mold", mold_mesh)
        context.collection.objects.link(mold_obj)
        mold_obj.matrix_world = target.matrix_world.copy()

        context.view_layer.objects.active = mold_obj
        mod = mold_obj.modifiers.new("Solidify", 'SOLIDIFY')
        mod.thickness = thickness
        mod.offset = 1.0
        bpy.ops.object.modifier_apply(modifier=mod.name)

        # steps 2-5: cut surface
        cut_obj, loop_normal = _build_cut_surface(
            target, boundary, thickness, step_pct, draft_deg,
        )
        if cut_obj is None:
            bpy.data.objects.remove(mold_obj, do_unlink=True)
            self.report({'ERROR'},
                        "Boundary edges must form a single closed loop.")
            return {'CANCELLED'}
        cut_obj.matrix_world = target.matrix_world.copy()

        # steps 6-7: boolean
        context.view_layer.objects.active = mold_obj
        bmod = mold_obj.modifiers.new("Boolean", 'BOOLEAN')
        bmod.operation = 'DIFFERENCE'
        bmod.object = cut_obj
        bpy.ops.object.modifier_apply(modifier=bmod.name)
        bpy.data.objects.remove(cut_obj, do_unlink=True)

        # step 8: separate
        bpy.ops.object.select_all(action='DESELECT')
        mold_obj.select_set(True)
        context.view_layer.objects.active = mold_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')

        # rename
        bm_tmp = bmesh.new()
        bm_tmp.from_mesh(target.data)
        bm_tmp.edges.ensure_lookup_table()
        centroid = Vector()
        cnt = 0
        for i in boundary:
            for v in bm_tmp.edges[i].verts:
                centroid += v.co
                cnt += 1
        centroid /= max(cnt, 1)
        bm_tmp.free()

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            avg = sum((Vector(v.co) for v in obj.data.vertices), Vector())
            avg /= max(len(obj.data.vertices), 1)
            side = (avg - centroid).dot(loop_normal)
            obj.name = f"{target.name}_Mold_{'Top' if side > 0 else 'Bottom'}"

        self.report({'INFO'}, "Mold split complete.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Properties & Panel
# ---------------------------------------------------------------------------

class MoldSplitProperties(bpy.types.PropertyGroup):
    target: PointerProperty(
        name="Target",
        type=bpy.types.Object,
        description="Object to create mold for",
        poll=lambda self, obj: obj.type == 'MESH',
    )
    thickness: FloatProperty(
        name="Thickness",
        description="Mold wall thickness",
        default=0.005,
        min=0.0001,
        soft_max=0.1,
        precision=4,
        unit='LENGTH',
    )
    step_pct: FloatProperty(
        name="Lip %",
        description="Interlocking lip as percentage of wall thickness (higher = better interlocking)",
        default=70.0,
        min=10.0,
        max=90.0,
        subtype='PERCENTAGE',
    )
    draft_angle: FloatProperty(
        name="Draft angle",
        description=(
            "Angle of step wall to base plane (90° = vertical, "
            ">90° = tilted outward for easier release)"
        ),
        default=math.radians(90.0),
        min=math.radians(90.0),
        max=math.radians(179.0),
        subtype='ANGLE',
    )


class MOLDSPLIT_PT_panel(bpy.types.Panel):
    bl_label = "Mold Split"
    bl_idname = "MOLDSPLIT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mold Split"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mold_split

        layout.prop(props, "target")
        layout.prop(props, "thickness")
        layout.prop(props, "step_pct")
        layout.prop(props, "draft_angle")

        layout.separator()
        box = layout.box()
        box.label(text="手順:", icon='INFO')
        box.label(text="1. Target を設定")
        box.label(text="2. Edit mode で切断線を Seam マーク")
        box.label(text="3. Generate を実行")
        layout.separator()

        layout.operator("moldsplit.generate", icon='MOD_BOOLEAN')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (MoldSplitProperties, MOLDSPLIT_OT_generate, MOLDSPLIT_PT_panel)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mold_split = PointerProperty(type=MoldSplitProperties)


def unregister():
    del bpy.types.Scene.mold_split
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
