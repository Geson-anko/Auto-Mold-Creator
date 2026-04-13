bl_info = {
    "name": "Mold Split",
    "author": "Auto-generated",
    "version": (2, 2, 0),
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
    """Polygon normal via Newell method.  Returns fallback Z-up for
    degenerate (collinear / coincident) input."""
    n = Vector()
    nv = len(positions)
    for i in range(nv):
        c, nx = positions[i], positions[(i + 1) % nv]
        n.x += (c.y - nx.y) * (c.z + nx.z)
        n.y += (c.z - nx.z) * (c.x + nx.x)
        n.z += (c.x - nx.x) * (c.y + nx.y)
    if n.length_squared < 1e-12:
        return Vector((0, 0, 1))
    n.normalize()
    return n


def _apply_modifier(context, obj, mod_name: str):
    """Select + activate *obj*, then apply modifier by name."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod_name)


def _build_cut_surface(context, target_obj, boundary_indices, thickness,
                       step_pct, draft_deg: float = 90.0):
    """Build a stepped cut surface using per-vertex miter offsets.

    4 layers per boundary vertex:
      L0 — original boundary position
      L1 — outward by step_h along mitered direction (step ledge)
      L2 — L1 + loop_normal * step_h (step wall top)
      L3 — extension flange far outward

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

    if loop_normal.dot(loop_centroid - mesh_center) < 0:
        loop_normal = -loop_normal

    bm.free()

    # per-edge outward direction
    edge_out: list[Vector] = []
    for i in range(num):
        j = (i + 1) % num
        ed = (positions[j] - positions[i]).normalized()
        out = loop_normal.cross(ed)
        mid = (positions[i] + positions[j]) * 0.5
        if out.dot(mid - loop_centroid) < 0:
            out = -out
        edge_out.append(out)

    # per-vertex mitered outward — single direction per vertex
    vmiter: list[Vector] = []
    for i in range(num):
        prev = (i - 1) % num
        avg = (edge_out[prev] + edge_out[i]).copy()
        avg.normalize()
        d = avg.dot(edge_out[i])
        vmiter.append(avg * (1.0 / max(d, 0.01)))

    # ------ build vertices (4 per boundary vert) ------
    cbm = bmesh.new()
    L0, L1, L2, L3 = [], [], [], []
    for j in range(num):
        p = positions[j]
        nm = loop_normal
        m = vmiter[j]

        l0 = p
        l1 = p + m * step_h
        l2 = l1 + nm * step_h + m * draft_off
        l3 = p + m * extension + nm * step_h + m * draft_off

        L0.append(cbm.verts.new(l0))
        L1.append(cbm.verts.new(l1))
        L2.append(cbm.verts.new(l2))
        L3.append(cbm.verts.new(l3))
    cbm.verts.ensure_lookup_table()

    # ------ faces ------
    # centre face
    cbm.faces.new(list(reversed(L0)))

    # per-edge quads (3 bands: step base, step wall, extension)
    for i in range(num):
        j = (i + 1) % num
        cbm.faces.new([L0[i], L0[j], L1[j], L1[i]])   # step base
        cbm.faces.new([L1[i], L1[j], L2[j], L2[i]])   # step wall
        cbm.faces.new([L2[i], L2[j], L3[j], L3[i]])   # extension flange

    mesh = bpy.data.meshes.new("_MoldSplit_Cut")
    cbm.to_mesh(mesh)
    cbm.free()

    bm2 = bmesh.new()
    bm2.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    bm2.to_mesh(mesh)
    bm2.free()

    obj = bpy.data.objects.new("_MoldSplit_Cut", mesh)
    context.collection.objects.link(obj)

    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
    mod.thickness = 1e-6
    mod.offset = -1.0
    _apply_modifier(context, obj, mod.name)

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

        mod = mold_obj.modifiers.new("Solidify", 'SOLIDIFY')
        mod.thickness = thickness
        mod.offset = 1.0
        _apply_modifier(context, mold_obj, mod.name)

        # steps 2-5: cut surface
        cut_obj, loop_normal = _build_cut_surface(
            context, target, boundary, thickness, step_pct, draft_deg,
        )
        if cut_obj is None:
            bpy.data.objects.remove(mold_obj, do_unlink=True)
            self.report({'ERROR'},
                        "Boundary edges must form a single closed loop.")
            return {'CANCELLED'}
        cut_obj.matrix_world = target.matrix_world.copy()

        # steps 6-7: boolean
        bmod = mold_obj.modifiers.new("Boolean", 'BOOLEAN')
        bmod.operation = 'DIFFERENCE'
        bmod.solver = 'EXACT'
        bmod.object = cut_obj
        _apply_modifier(context, mold_obj, bmod.name)
        bpy.data.objects.remove(cut_obj, do_unlink=True)

        # check boolean produced usable geometry
        if len(mold_obj.data.vertices) == 0:
            bpy.data.objects.remove(mold_obj, do_unlink=True)
            self.report({'ERROR'},
                        "Boolean failed — check mesh is manifold and "
                        "seam loop fully divides the surface.")
            return {'CANCELLED'}

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
        description="Interlocking lip as percentage of wall thickness "
                    "(higher = better interlocking)",
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
