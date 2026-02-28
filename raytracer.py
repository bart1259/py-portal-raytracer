import numpy as np
import pyopencl as cl

import os
import sys

import pygame
import pygame.font

import time

WIDTH = 768
HEIGHT = 432

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
surf = pygame.Surface((WIDTH, HEIGHT))

# Set title
pygame.display.set_caption("Portal Raytracer Demo")
small_font = pygame.font.Font("FreeSans.ttf", 16)

os.environ["PYOPENCL_CTX"] = "0"

OBJ_FILE = "world.obj"

with open(OBJ_FILE, "r") as f:
    lines = f.readlines()

verticies = []
indices = []
uvs = []
uv_indicies = []
for line in lines:
    if line.startswith("v "):
        # Vertex position
        parts = line.split()[1:]
        verticies.append([float(p) for p in parts] + [1.0])  # Homogeneous coordinate
    elif line.startswith("f "):
        # Face (triangle) definition
        parts = line.split()[1:]
        face = [int(p.split("/")[0]) - 1 for p in parts]  # Convert to 0-based index
        indices.append(face)

        try:
            uv_face = [int(p.split("/")[1]) - 1 for p in parts if "/" in p]  # UV indices
            uv_indicies.append(uv_face)
        except:
            pass # No UVs on this mesh
    elif line.startswith("vt "):
        # UV coordinates
        parts = line.split()[1:]
        uvs.append([float(parts[0]), float(parts[1])])

if len(uvs) == 0:
    # If no UVs are defined, create a default UV mapping
    uvs = [[0.0, 0.0]] * len(verticies)  # Default UVs for each vertex
    uv_indicies = [[0, 0, 0]] * len(indices)  # Default UV indices for each face

verticies = np.array(verticies, dtype=np.float32)
verticies = verticies.reshape(-1, 4)  # Ensure it's a 2D array with 4 columns

indices = np.array(indices, dtype=np.uint32)
indices = indices.reshape(-1, 3)

uvs = np.array(uvs, dtype=np.float32)
uvs = uvs.reshape(-1, 2)  # Ensure it's a 2D array with 2 columns

uv_indicies = np.array(uv_indicies, dtype=np.uint32)

print(f"Loaded {len(verticies)} vertices and {len(indices)} indices from {OBJ_FILE}")

class BoundingBox:
    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def get_surface_area(self):
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        depth = self.max_z - self.min_z
        return 2 * (width * height + width * depth + height * depth)

class Triangle:
    def __init__(self, index, v0, v1, v2):
        self.original_mesh_index = index
        self.index = index
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def get_bounding_box(self):
        min_x = min(self.v0[0], self.v1[0], self.v2[0])
        min_y = min(self.v0[1], self.v1[1], self.v2[1])
        min_z = min(self.v0[2], self.v1[2], self.v2[2])
        max_x = max(self.v0[0], self.v1[0], self.v2[0])
        max_y = max(self.v0[1], self.v1[1], self.v2[1])
        max_z = max(self.v0[2], self.v1[2], self.v2[2])
        return BoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)
    
class TriangleList:
    def __init__(self, triangles):
        self.triangles = triangles

    def get_bounding_box(self):
        if not self.triangles:
            return None
        
        # initialize mins/maxs
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        # sweep over every vertex of every triangle
        for tri in self.triangles:
            for v in (tri.v0, tri.v1, tri.v2):
                x, y, z = v.tolist()[:3] 
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if z < min_z: min_z = z
                if x > max_x: max_x = x
                if y > max_y: max_y = y
                if z > max_z: max_z = z

        return BoundingBox(min_x, min_y, min_z,
                           max_x, max_y, max_z)
    
    def split(self):
        MAX_TRIANGLES_PER_LEAF = 4
        if len(self.triangles) <= MAX_TRIANGLES_PER_LEAF:
            return None, None # Indicate no split

        if len(self.triangles) == 2:
            return TriangleList([self.triangles[0]]), TriangleList([self.triangles[1]])

        centers = np.array([[(triangle.v0[0] + triangle.v1[0] + triangle.v2[0]) / 3.0,
                             (triangle.v0[1] + triangle.v1[1] + triangle.v2[1]) / 3.0,
                             (triangle.v0[2] + triangle.v1[2] + triangle.v2[2]) / 3.0] for triangle in self.triangles], dtype=np.float32)


        # test to see if x or y or z is the best axis to split
        best_axis = None
        best_split_point = None
        min_loss = float('inf')

        BIN_COUNT = 8

        for axis in range(3):
            split_points = np.linspace(np.min(centers[:, axis]), np.max(centers[:, axis]), num=BIN_COUNT)
            split_points = split_points[1:-1]  # Exclude the first and last points to avoid trivial splits
            for split_point in split_points:
                affiliations = centers[:, axis] > split_point

                if np.sum(affiliations) == 0 or np.sum(affiliations) == affiliations.shape[0]:
                    # There is no split, skip
                    continue

                tris_left = [self.triangles[i] for i in range(len(self.triangles)) if not affiliations[i]]
                tris_right = [self.triangles[i] for i in range(len(self.triangles)) if affiliations[i]]
                left_bb = TriangleList(tris_left).get_bounding_box()
                right_bb = TriangleList(tris_right).get_bounding_box()

                loss = (len(tris_left) * left_bb.get_surface_area()) + (len(tris_right) * right_bb.get_surface_area())

                if loss < min_loss:
                    best_axis = axis
                    best_split_point = split_point
                    min_loss = loss

        # Split based on the best axis
        split_affiliations = centers[:, best_axis] > best_split_point

        left_triangles = []
        right_triangles = []
        for i, triangle in enumerate(self.triangles):
            if split_affiliations[i]:
                right_triangles.append(triangle)
            else:
                left_triangles.append(triangle)

        left = TriangleList(left_triangles)
        right = TriangleList(right_triangles)

        return left, right
    
    def __len__(self):
        return len(self.triangles)
    
    def __getitem__(self, index):
        return self.triangles[index]

bvh_type = np.dtype([("bb_min_x", np.float32), ("bb_min_y", np.float32), ("bb_min_z", np.float32), ("padding1", np.float32),  # Padding for alignment
                     ("bb_max_x", np.float32), ("bb_max_y", np.float32), ("bb_max_z", np.float32), ("padding2", np.float32),  # Padding for alignment
                     ("left_child", np.int32), ("right_child", np.int32),
                     ("triangle_start", np.int32), ("triangle_count", np.int32),
                     ("depth", np.int32)])  # Padding for alignment  

class BVHNode:
    # Struct packing
    # bb_min_x, bb_min_y, bb_min_z, 12 bytes
    # bb_max_x, bb_max_y, bb_max_z, 12 bytes
    # left_child, right_child       8 bytes
    # triangle_start, triangle_count, 8 bytes
    # total: 40 bytes
    instance_count = 0
    def __init__(self, bounding_box, left=None, right=None, triangles=[], depth=0):
        self.id = BVHNode.instance_count
        BVHNode.instance_count += 1
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.triangles = triangles
        self.depth = depth

    def is_leaf(self):
        return len(self.triangles) > 0
    
    def print_tree(self):
        print(f"Node: {self.bounding_box.min_x}, {self.bounding_box.min_y}, {self.bounding_box.min_z} -> {self.bounding_box.max_x}, {self.bounding_box.max_y}, {self.bounding_box.max_z}")
        if not self.is_leaf():
            if self.left:
                print("Left child:")
                self.left.print_tree()

    def order_triangles(self, start_index=0):
        if self.is_leaf():
            for tri in self.triangles:
                tri.index = start_index
                start_index += 1
            return self.triangles
        triangles = []
        if self.left:
            triangles.extend(self.left.order_triangles(start_index))
        if self.right:
            triangles.extend(self.right.order_triangles(start_index + len(triangles)))
        return triangles
    
    def get_packed_struct(self):
        bb_min_x = self.bounding_box.min_x
        bb_min_y = self.bounding_box.min_y
        bb_min_z = self.bounding_box.min_z
        bb_max_x = self.bounding_box.max_x
        bb_max_y = self.bounding_box.max_y
        bb_max_z = self.bounding_box.max_z
        
        left_child = -1 if not self.left else self.left.id
        right_child = -1 if not self.right else self.right.id
        
        triangle_start = -1 if not self.is_leaf() else self.triangles[0].index
        triangle_count = 0 if not self.is_leaf() else len(self.triangles)

        return np.array((bb_min_x, bb_min_y, bb_min_z, 0, # padding for alignment
                          bb_max_x, bb_max_y, bb_max_z, 0, # padding for alignment
                         left_child, right_child, triangle_start, triangle_count,
                         self.depth), dtype=bvh_type)
    
    def __repr__(self):
        return f"BVHNode(id={self.id}, bounding_box=({self.bounding_box.min_x}, {self.bounding_box.min_y}, {self.bounding_box.min_z}) -> ({self.bounding_box.max_x}, {self.bounding_box.max_y}, {self.bounding_box.max_z}), left={self.left}, right={self.right}, triangles={self.triangle})"

def get_packed_struct_array(root):
    stack = [root]
    packed_structs = [None] * BVHNode.instance_count
    while stack:
        node = stack.pop()
        packed_structs[node.id] = node.get_packed_struct()
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return packed_structs

def construct_bvh_tree(triangle_list, depth=0):
    root = BVHNode(triangle_list.get_bounding_box(), depth=depth)
    left, right = triangle_list.split()

    if left is None and right is None:
        # The algorithm decided not to split, create a leaf node
        root.triangles = triangle_list
        root.bounding_box = triangle_list.get_bounding_box()
        return root
    else:
        left_bvh = None
        if len(left) == 1: # Not sure if this is needed anymore since > 1 triangles are handled in the split method
            left_bvh = BVHNode(left[0].get_bounding_box(), triangles=[left[0]], depth=depth+1)
        else:
            left_bvh = construct_bvh_tree(left, depth=depth+1)

        right_bvh = None
        if len(right) == 1:
            right_bvh = BVHNode(right[0].get_bounding_box(), triangles=[right[0]], depth=depth+1)
        else:
            right_bvh = construct_bvh_tree(right, depth=depth+1)

        root.left = left_bvh
        root.right = right_bvh
        return root

triangles = []
for i in range(len(indices)):
    v0 = verticies[indices[i][0]]
    v1 = verticies[indices[i][1]]
    v2 = verticies[indices[i][2]]
    triangles.append(Triangle(i, v0, v1, v2))

triangle_list = TriangleList(triangles)
start_time = time.time()

bvh = construct_bvh_tree(triangle_list)

# Reorder the mesh triangles so ones in the same node are together
ordered_triangles = bvh.order_triangles()

new_orders = np.array([tri.original_mesh_index for tri in ordered_triangles], dtype=np.uint32)
indices = indices[new_orders]
uv_indicies = uv_indicies[new_orders]

structs = get_packed_struct_array(bvh)
bvh_structs = np.array(structs, dtype=bvh_type)

end_time = time.time()
print(f"BVH construction took {end_time - start_time:.2f} seconds")


portal_type = np.dtype([
    ("v0", np.float32, (4,)), 
    ("v1", np.float32, (4,)), 
    ("v2", np.float32, (4,)), 
    ("v3", np.float32, (4,)), 
    ("ray_transform", np.float32, (16,))
])

def make_frame(verts):
    """
    verts: list/array of 4 points (v0,v1,v2,v3),
    ordered as lower-left, lower-right, upper-right, upper-left
    Returns 4x4 matrix that maps local coords [x,y,z,1] -> world [X,Y,Z,1].
    """
    v0, v1, v2, v3 = [np.array(v, dtype=float) for v in verts]
    # x axis: from v0 to v1
    x = v1 - v0
    x /= np.linalg.norm(x)
    # y axis: from v0 to v3
    y = v3 - v0
    y /= np.linalg.norm(y)
    # z axis: orthonormal
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    # Re-orthogonalize y in case input wasn't perfect
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    # Build 4×4
    M = np.eye(4)
    M[0:3,0] = x
    M[0:3,1] = y
    M[0:3,2] = z
    M[0:3,3] = v0
    return M

class Portal:
    def __init__(self, from_verts, to_verts):
        self.from_verts = np.array(from_verts, dtype=np.float32)
        self.to_verts = np.array(to_verts, dtype=np.float32)
        # build local->world for each portal
        self.Mf = make_frame(from_verts)
        self.Mt = make_frame(to_verts)
        # invert world->local for from
        self.Mf_inv = np.linalg.inv(self.Mf)
        # one‐way warp
        self.T = self.Mt @ self.Mf_inv
        # if you want backward warp:
        self.Tinv = self.Mf @ np.linalg.inv(self.Mt)

    def warp_point(self, p):
        ph = np.append(p, 1.0)
        ph2 = self.T @ ph
        return (ph2[:3] / ph2[3]).astype(np.float32)

    def warp_dir(self, d):
        dh = np.append(d, 0.0)
        dh2 = self.T @ dh
        return (dh2[:3]).astype(np.float32)

    def get_packed_struct(self):
        values = (
            self.from_verts[0].tolist() + [0],
            self.from_verts[1].tolist() + [0],
            self.from_verts[2].tolist() + [0],
            self.from_verts[3].tolist() + [0],
            self.T.flatten().tolist()
        )
        
        return np.array(values, dtype=portal_type)
    
    def ray_triangle_intersection(self, ray_origin, ray_direction, v0, v1, v2):
        # Moller–Trumbore algorithm
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)

        if abs(a) < 1e-8:
            return -1
        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return -1
        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return -1
        t = f * np.dot(edge2, q)
        if t > 1e-8:  # Intersection occurs
            return t
        else:
            return -1
    
    def is_ray_through_portal(self, ray_origin, ray_direction, ray_distance):
        # Given a ray and max distance, check if it goes through the portal.
        t = self.ray_triangle_intersection(ray_origin, ray_direction, portal.from_verts[0], portal.from_verts[1], portal.from_verts[2])
        if t >= 0 and t <= ray_distance:
            return True
        t = self.ray_triangle_intersection(ray_origin, ray_direction, portal.from_verts[1], portal.from_verts[2], portal.from_verts[3])
        if t >= 0 and t <= ray_distance:
            return True
        return False


def load_portals(portal_file="world.portals"):
    with open(portal_file, "r") as f:
        lines = f.readlines()

    portals = {}
    portal_name = None
    portal_verts = []
    for line in lines:
        if line.startswith("portal "):
            portal_name = line.split()[1].strip()
            portal_verts = []

        elif line.startswith("0: ") or line.startswith("1: ") or line.startswith("2: ") or line.startswith("3: "):
            parts = line.split(":")[1].strip().split(",")
            vertex = [float(p) for p in parts]
            portal_verts.append(vertex)

        elif line.startswith("endportal"):
            if portal_name and portal_verts:
                portal_index = portal_name[:-1]
                if portal_index not in portals:
                    portals[portal_index] = {"a": None, "b": None}
                
                portals[portal_index][portal_name[-1]] = portal_verts

            portal_name = None
            portal_verts = []


    portal_list = []
    for portal_index, portal_data in portals.items():
        if portal_data["a"] is not None and portal_data["b"] is not None:
            portal = Portal(portal_data["a"], portal_data["b"])
            portal_list.append(portal)

            portal = Portal(portal_data["b"], portal_data["a"])  # Add the reverse portal
            portal_list.append(portal)

    return portal_list

portal_structs = []
portals = load_portals("world.portals")

for i, portal in enumerate(portals):
    portal_structs.append(portal.get_packed_struct())

portal_structs = np.array(portal_structs, dtype=portal_type)

## Load image
from PIL import Image

img = Image.open("World.png").convert("RGBA")
width, height = img.size
img_np = np.array(img).astype(np.uint8)    # shape = (height, width, 4)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
device_name = ctx.devices[0].name

mf = cl.mem_flags
verts_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=verticies)
indices_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices)
uvs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uvs)
uv_indicies_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uv_indicies)

bvh_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bvh_structs)
portal_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=portal_structs)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, WIDTH * HEIGHT * 4 * 4) # 4 bytes per float, 3 floats per pixel, (aligned to 4 bytes)

fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
cl_image = cl.create_image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, hostbuf=img_np, shape=(width, height))
sampler = cl.Sampler(ctx, True, cl.addressing_mode.CLAMP_TO_EDGE, cl.filter_mode.NEAREST)

with open("kernel.cl", "r") as f:
    kernel_code = f.read()

prg = cl.Program(ctx, kernel_code).build()

knl = prg.render  # Get a handle to the kernel


def render(camera_position, camera_angle=0.0, bvh_debug_depth=0):
    camera_fov = np.float32(60.0)
    camera_angle = np.float32(camera_angle)
    camera_position = np.float32([camera_position[0], camera_position[1], camera_position[2], 0.0])
    triangle_count = np.int32(len(indices))  # Number of triangles
    bvh_debug_depth = np.int32(bvh_debug_depth)  # Number of BVH nodes
    portal_count = np.int32(len(portals))  # Number of portals

    res_np = np.empty((HEIGHT, WIDTH, 4), dtype=np.float32) # Kernel output buffer
    knl(queue, (WIDTH, HEIGHT), None, camera_fov, camera_position, camera_angle, bvh_debug_depth, verts_g, indices_g, uvs_g, uv_indicies_g, bvh_g, portal_g, triangle_count, portal_count, cl_image, sampler, res_g)
    cl.enqueue_copy(queue, res_np, res_g)
    queue.finish()


    res_np = res_np.transpose((1, 0, 2))  # Ensure the output is in the correct shape
    return res_np

# camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
camera_pos = np.array([0.0, -0.9000002, 5.0], dtype=np.float32)
camera_angle = 0.0
bvh_debug_depth = 0
key_state = {}
fps = []

frame = 0
dt = 0
SPEED = 5.0 

while True:
    for evt in pygame.event.get():
        if evt.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif evt.type == pygame.KEYDOWN:
            if evt.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            else:
                key_state[evt.key] = True
        elif evt.type == pygame.KEYUP:
            key_state[evt.key] = False

    prev_camera_pos = camera_pos.copy()

    speed = SPEED

    if key_state.get(pygame.K_LSHIFT) or key_state.get(pygame.K_RSHIFT):
        speed *= 2.0

    if key_state.get(pygame.K_w):
        camera_pos[0] += speed * dt * np.sin(camera_angle)
        camera_pos[2] -= speed * dt * np.cos(camera_angle)
    if key_state.get(pygame.K_s):
        camera_pos[0] -= speed * dt * np.sin(camera_angle)
        camera_pos[2] += speed * dt * np.cos(camera_angle)
    if key_state.get(pygame.K_a):
        camera_pos[0] -= speed * dt * np.cos(camera_angle)
        camera_pos[2] -= speed * dt * np.sin(camera_angle)
    if key_state.get(pygame.K_d):
        camera_pos[0] += speed * dt * np.cos(camera_angle)
        camera_pos[2] += speed * dt * np.sin(camera_angle)
    if key_state.get(pygame.K_e):
        camera_pos[1] -= speed * dt
    if key_state.get(pygame.K_q):
        camera_pos[1] += speed * dt

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    mouse_x, mouse_y = pygame.mouse.get_pos()
    if pygame.mouse.get_focused():
        # Rotate camera based on mouse movement
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        camera_angle += mouse_dx * 0.005

    # Check if we went through a portal
    for portal in portals:
        camera_dir = (camera_pos - prev_camera_pos) / np.linalg.norm(camera_pos - prev_camera_pos)
        if portal.is_ray_through_portal(prev_camera_pos, camera_dir, np.linalg.norm(camera_pos - prev_camera_pos)):
            camera_pos = portal.warp_point(camera_pos)
            dir = np.array([np.sin(camera_angle), 0.0, -np.cos(camera_angle)], dtype=np.float32)
            new_dir = portal.warp_dir(dir)
            camera_angle = np.arctan2(new_dir[0], -new_dir[2])  # Adjust angle based on new direction
            break

    numbers = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]
    for i, num in enumerate(numbers):
        if key_state.get(num):
            bvh_debug_depth = i + 1
            if key_state.get(pygame.K_LSHIFT) or key_state.get(pygame.K_RSHIFT):
                bvh_debug_depth += 10

    if key_state.get(pygame.K_MINUS):
        bvh_debug_depth = 0

    start_time = time.time()
    res_np = render(camera_position=camera_pos, camera_angle=camera_angle, bvh_debug_depth=bvh_debug_depth)
    
    res_np = (res_np * 255).astype(np.uint8)  # Ensure the array is in the correct format for pygame

    end_time = time.time()
    dt = end_time - start_time

    # Display FPS Text in upper left corner
    fps.append(1 / dt)
    fps = fps[-60:]  # Keep only the last 60 frames
    avg_fps = sum(fps) / len(fps) if fps else 0
    fps_text = small_font.render(f"FPS: {avg_fps:.0f}", True, (255, 255, 255))
    device_text = small_font.render(f"{device_name} ({WIDTH}x{HEIGHT})", True, (255, 255, 255))

    # Pygame render
    pygame.surfarray.blit_array(surf, res_np[:, :, :3])
    screen.blit(surf, (0, 0))
    screen.blit(device_text, (10, 10))
    screen.blit(fps_text, (10, 32))
    pygame.display.flip()
