bl_info = {
    "name": "Portal Test",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Object menu; File > Export"
}

import bpy
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty
from bpy import context
import json

class OBJECT_OT_link_portals(bpy.types.Operator):
    bl_idname = "object.link_portals"
    bl_label = "Link Portals"
    bl_description = "Links two quads as portals"

    def execute(self, context):
        if len(context.selected_objects) != 2:
            raise "Must be exactly two sepected objects"

        for obj in context.selected_objects:
            print(obj.name)
            

        # portal_data_raw = bpy.data.texts.load(".portals")
        # bpy.data.texts.remove(portal_data_raw)

        tb = bpy.data.texts.get(".portals")
        if tb is None:
            tb = bpy.data.texts.new(name=".portals")
            tb.write(json.dumps({"portal_count": 1}))

        data = json.loads(tb.as_string())
        count = data["portal_count"]

        objs = context.selected_objects
        objs[0].name = f"portal_{count}a"
        objs[1].name = f"portal_{count}b"

        data["portal_count"] += 1
        tb.clear()  # remove old contents
        tb.write(json.dumps(data))
        
        print("Done Linking Portals!")
        return {'FINISHED'}


def menu_func_link_portals(self, context):
    # add to the Object menu in the 3D View
    self.layout.operator(OBJECT_OT_link_portals.bl_idname, icon='INFO')


class EXPORT_MESH_OT_portals(bpy.types.Operator, ExportHelper):
    bl_idname = "export_mesh.portals"
    bl_label = "Export Portals"
    bl_description = "Export vertices of portals named portal_*[a/b] to a .portals file"

    # ExportHelper mixin class uses this
    filename_ext = ".portals"

    filter_glob: StringProperty(
        default="*.portals",
        options={'HIDDEN'},
    )

    def execute(self, context):
        # self.filepath is set by ExportHelper
        count = 0
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            for ob in context.scene.objects:
                if ob.type == 'MESH' and ob.name.startswith("portal_"):
                    mesh = ob.data
                    f.write(f"portal {ob.name.split('_')[1]}\n")
                    for i, v in enumerate(mesh.vertices):
                        co_world = ob.matrix_world @ v.co
                        f.write(f"{i}: {co_world.x:.6f}, {co_world.z:.6f}, {-co_world.y:.6f}\n")
                    count += 1
                    f.write("endportal\n")
                    f.write("\n")

        self.report({'INFO'}, f"Exported {count} object(s) to {self.filepath}")
        return {'FINISHED'}


def menu_func_export(self, context):
    # add to File > Export menu
    self.layout.operator(EXPORT_MESH_OT_portals.bl_idname, text="Export Portals (.portals)")


classes = (
    OBJECT_OT_link_portals,
    EXPORT_MESH_OT_portals,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # hook into menus
    bpy.types.VIEW3D_MT_object.append(menu_func_link_portals)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    # remove menu items
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.VIEW3D_MT_object.remove(menu_func_link_portals)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()