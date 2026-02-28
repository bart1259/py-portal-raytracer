#define DEBUG 0

typedef struct __attribute__((packed)) bvh_struct {
    float3 aabb_min;  
    float3 aabb_max;  
    int left_child;
    int right_child;
    int triangle_start;
    int triangle_count;
    int depth;
} bvh_struct_t;

typedef struct portal {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 v3;
    float16 ray_transform; // Transformation matrix for the ray
} portal_t;

inline float4 mat4_mul_vec4(const float16 M, const float4 v)
{
    return (float4)(
        M.s0 * v.x + M.s1 * v.y + M.s2 * v.z + M.s3 * v.w,
        M.s4 * v.x + M.s5 * v.y + M.s6 * v.z + M.s7 * v.w,
        M.s8 * v.x + M.s9 * v.y + M.sa * v.z + M.sb * v.w,
        M.sc * v.x + M.sd * v.y + M.se * v.z + M.sf * v.w
    );
}

inline void ray_aabb_intersection(
   float3 ray_origin, 
   float3 ray_inv_direction,
   float3 aabb_min, 
   float3 aabb_max,
   __private float* t_out
){
    *t_out = -1.0f; // Initialize to no intersection
    // t0 = (mn - ro) * invd;  t1 = (mx - ro) * invd;
    float3 t0 = (aabb_min - ray_origin) * ray_inv_direction;
    float3 t1 = (aabb_max - ray_origin) * ray_inv_direction;
    // tmin = max( max(min(t0.x,t1.x), min(t0.y,t1.y)), min(t0.z,t1.z) );
    float3 tmin3 = fmin(t0, t1);
    float3 tmax3 = fmax(t0, t1);
    float  tmin  = fmax( fmax(tmin3.x, tmin3.y), tmin3.z );
    float  tmax  = fmin( fmin(tmax3.x, tmax3.y), tmax3.z );
    if( tmax >= fmax(tmin, 0.0f)) {
        *t_out = tmin < 0.0f ? 0.0f : tmin;
    }
}

void ray_triangle_intersection(
    float3 ray_origin, float3 ray_direction,
    float3 v0, float3 v1, float3 v2,
    float* out_t,
    float* out_u,
    float* out_v
)
{
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(ray_direction, edge2);
    float a = dot(edge1, h);
    if (a > -1e-8 && a < 1e-8) {
        *out_t = -1.0f; // Ray is parallel to the triangle
        return;
    }
    float f = 1.0f / a;
    float3 s = ray_origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return; // Intersection is outside the triangle
    float3 q = cross(s, edge1);
    float v = f * dot(ray_direction, q);
    if (v < 0.0f || u + v > 1.0f)
        return; // Intersection is outside the triangle
    float t_value = f * dot(edge2, q);
    if (t_value > 1e-8) {
        *out_t = t_value; // Intersection found
        if (out_u) *out_u = u;
        if (out_v) *out_v = v;
    } else {
        *out_t = -1.0f; // Intersection is behind the ray origin
    }
    return;                 
}

void ray_trace_bvh(
    float3 ray_origin,
    float3 ray_direction,
    __global float3* triangle_verticies_g,
    __global int* triangle_indicies_g,
    __global float2* uvs_g,
    __global int* triangle_uv_indicies_g,
    __global bvh_struct_t* bvh_structs_g,
    int triangle_count,
    float* out_t, // Output parameter for intersection distance
    float3* out_position, // Output parameter for intersection position
    float3* out_normal, // Output parameter for intersection normal
    float2* out_uv, // Output parameter for intersection UVs
    int bvh_debug_depth // Debug depth for BVH traversal
#ifdef DEBUG
    , float3* debug_pixel_color // Used to output BVH data
#endif
)
{
    float3 ray_inv_direction = (float3)(
        ray_direction.x != 0.0f ? 1.0f / ray_direction.x : INFINITY,
        ray_direction.y != 0.0f ? 1.0f / ray_direction.y : INFINITY,
        ray_direction.z != 0.0f ? 1.0f / ray_direction.z : INFINITY
    );

    uint3 ray_sign_mask = (uint3)(
        ray_direction.x < 0.0f ? 1.0f : 0.0f,
        ray_direction.y < 0.0f ? 1.0f : 0.0f,
        ray_direction.z < 0.0f ? 1.0f : 0.0f
    );

#ifdef DEBUG
    float3 DEBUG_COLORS[] = {
        (float3)(1.0f, 0.0f, 0.0f),   // Red
        (float3)(0.0f, 1.0f, 0.0f),   // Green
        (float3)(0.0f, 0.0f, 1.0f),   // Blue
        (float3)(1.0f, 1.0f, 0.0f),   // Yellow
        (float3)(1.0f, 0.5f, 0.5f),   // Light Red
        (float3)(0.5f, 1.0f, 0.5f),   // Light Green
        (float3)(0.5f, 0.5f, 1.0f),   // Light Blue
        (float3)(1.0f, 0.5f, 1.0f),   // Light Magenta
        (float3)(0.5f, 1.0f, 1.0f),   // Light Cyan
        (float3)(0.5f, 0.5f, 0.5f),   // Gray
        (float3)(1.0f, 1.0f, 1.0f),   // White
        (float3)(1.0f, 0.65f, 0.0f),  // Orange
        (float3)(0.6f, 0.4f, 0.2f),   // Brown
        (float3)(0.2f, 0.2f, 0.2f),   // Dark Gray
        (float3)(0.8f, 0.0f, 0.5f),   // Magenta
        (float3)(0.5f, 0.0f, 0.5f),   // Purple
        (float3)(0.75f, 0.75f, 0.0f), // Olive
        (float3)(0.0f, 0.5f, 0.5f),   // Teal
        (float3)(0.0f, 0.75f, 0.25f), // Spring Green
        (float3)(0.5f, 0.0f, 0.0f),   // Dark Red
        (float3)(0.0f, 0.5f, 0.0f),   // Dark Green
        (float3)(0.0f, 0.0f, 0.5f),   // Dark Blue
        (float3)(0.2f, 0.6f, 1.0f),   // Sky Blue
        (float3)(1.0f, 0.75f, 0.8f),  // Pink
        (float3)(0.75f, 0.0f, 0.25f), // Crimson
        (float3)(0.4f, 0.2f, 0.6f),   // Indigo
        (float3)(0.9f, 0.9f, 0.5f),   // Light Yellow
        (float3)(0.3f, 0.1f, 0.0f),   // Maroon
        (float3)(0.1f, 0.0f, 0.3f),   // Navy
        (float3)(0.9f, 0.3f, 0.2f)    // Coral
    };
    int COLOR_COUNT = 30;
#endif

    uint stack[16];
    int stack_ptr = 0;

    stack[stack_ptr++] = 0; // Start with the root node

    float min_t = INFINITY; // Initialize to a large value

    int triangle_tests = 0;
    int aabb_tests = 0;

    float hit_u;
    float hit_v;

    while (stack_ptr != 0)
    {
        // Check depth of current node
        int index = stack[stack_ptr - 1];
        bvh_struct_t current_node = bvh_structs_g[index];
        stack_ptr--; // Pop the current node

        if (bvh_debug_depth != 0 && current_node.depth > bvh_debug_depth) {
            // If the node is too deep, skip it
            continue;
        }

        // Check if the ray intersects the AABB of the current node
        float hit_t = -1.0f;

        ray_aabb_intersection(
            ray_origin, 
            ray_inv_direction,
            current_node.aabb_min, 
            current_node.aabb_max, 
            &hit_t
        );
        aabb_tests++;


        if (hit_t < 0.0f) {
            // No intersection with this node's AABB, skip it
            continue;
        } else if(hit_t > min_t) {
            // If the intersection is further than the closest hit, skip it
            continue;
        } 
        else {
            if (current_node.left_child == -1 && bvh_debug_depth == 0) {
                // Check for triangle intersection
                int triangle_start = current_node.triangle_start;
                for (int i = 0; i < current_node.triangle_count; i++) {
                    int triangle_index = triangle_start + i;

                    int i0 = triangle_indicies_g[triangle_index*3 + 0];
                    int i1 = triangle_indicies_g[triangle_index*3 + 1];
                    int i2 = triangle_indicies_g[triangle_index*3 + 2];
                    
                    float3 hit_normal = normalize(cross(
                        triangle_verticies_g[i1] - triangle_verticies_g[i0],
                        triangle_verticies_g[i2] - triangle_verticies_g[i0]
                    ));

                    // Backface culling
                    if (dot(hit_normal, ray_direction) > 0.0f) {
                        continue;
                    }

                    float hit_dist = -1.0f;
                    ray_triangle_intersection(
                        ray_origin,
                        ray_direction,
                        triangle_verticies_g[i0],
                        triangle_verticies_g[i1],
                        triangle_verticies_g[i2],
                        &hit_dist,
                        &hit_u,
                        &hit_v
                    );

                    triangle_tests++;

                    if (hit_dist >= 0.0f && hit_dist < min_t) {
                        min_t = hit_dist; // Update min_t

                        // Update normals
                        *out_normal = hit_normal;

                        // Update UV
                        float2 uv0 = uvs_g[triangle_uv_indicies_g[triangle_index*3 + 0]];
                        float2 uv1 = uvs_g[triangle_uv_indicies_g[triangle_index*3 + 1]];
                        float2 uv2 = uvs_g[triangle_uv_indicies_g[triangle_index*3 + 2]];

                        float w = 1.0f - hit_u - hit_v;
                        float2 uv = (uv0 * w) + (uv1 * hit_u) + (uv2 * hit_v);
                        *out_uv = uv;
                    }
                }

            } else {
                bvh_struct_t left_node = bvh_structs_g[current_node.left_child];
                bvh_struct_t right_node = bvh_structs_g[current_node.right_child];

                float left_hit_t = -1.0f;
                float right_hit_t = -1.0f;

                ray_aabb_intersection(
                    ray_origin, 
                    ray_inv_direction,
                    left_node.aabb_min, 
                    left_node.aabb_max, 
                    &left_hit_t
                );

                ray_aabb_intersection(
                    ray_origin, 
                    ray_inv_direction,
                    right_node.aabb_min, 
                    right_node.aabb_max, 
                    &right_hit_t
                );

                aabb_tests += 2;

                if (left_hit_t < 0.0f && right_hit_t < 0.0f) {
                    continue;
                } else if (left_hit_t < 0.0f) {
                    stack[stack_ptr++] = current_node.right_child;
                } else if (right_hit_t < 0.0f) {
                    stack[stack_ptr++] = current_node.left_child;
                } else if (left_hit_t < right_hit_t) {
                    stack[stack_ptr++] = current_node.right_child;
                    stack[stack_ptr++] = current_node.left_child;
                } else {
                    stack[stack_ptr++] = current_node.left_child;
                    stack[stack_ptr++] = current_node.right_child;
                }

                if (bvh_debug_depth != 0 && current_node.depth == bvh_debug_depth && hit_t < min_t) {
                    *debug_pixel_color = (float3)DEBUG_COLORS[index % COLOR_COUNT]; // Use triangle_start as an index for color 
                    min_t = hit_t; // Update min_t
                }
            }
        }
    }

    int MAX_AABB_TESTS = 50;
    int MAX_TRIANGLE_TESTS = 25;
    float aabb_value = aabb_tests / (float)MAX_AABB_TESTS;
    float triangle_value = triangle_tests / (float)MAX_TRIANGLE_TESTS;

    aabb_value = clamp(aabb_value, 0.0f, 1.0f); // Clamp value to [0, 1]
    triangle_value = clamp(triangle_value, 0.0f, 1.0f); // Clamp value to [0, 1]

    if (min_t >= INFINITY) {
        // No intersection found
        *out_t = -1.0f;
        *out_position = (float3)(0.0f, 0.0f, 0.0f);
        *out_normal = (float3)(0.0f, 0.0f, 0.0f);
        return;
    }

    *out_t = min_t;
    *out_position = ray_origin + ray_direction * min_t;

}

void check_portal_intersection(
    float3 ray_origin,
    float3 ray_direction,
    portal_t portal,
    float* hit_dist
)
{
    // Check intersection with the first triangle of the portal
    ray_triangle_intersection(
        ray_origin,
        ray_direction,
        portal.v0,
        portal.v1,
        portal.v2,
        hit_dist,
        NULL,
        NULL
    );

    if (*hit_dist < 0.0f) {
        // If no intersection, check the second triangle
        ray_triangle_intersection(
            ray_origin,
            ray_direction,
            portal.v1,
            portal.v2,
            portal.v3,
            hit_dist,
            NULL,
            NULL
        );
    }
}

__kernel void render(
    float camera_fov, 
    float4 camera_origin,
    float camera_angle,
    int bvh_debug_depth,
    __global float3* triangle_verticies_g,
    __global int* triangle_indicies_g,
    __global float2* triangle_uvs_g,
    __global int* triangle_uv_indicies_g,
    __global bvh_struct_t* bvh_structs_g,
    __global portal_t* portal_g,
    int triangle_count,
    int portal_count,
    __read_only image2d_t world_texture,
    sampler_t world_texture_sampler,
    __global float4* res_g
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = get_global_size(0);
    int height = get_global_size(1);

    // Get the direction the ray is pointing based on fov
    float aspect_ratio = (float)width / (float)height;
    float fov_rad = camera_fov * (3.14159265358979323846f / 180.0f);
    float tan_half_fov = tan(fov_rad / 2.0f);
    float x_ndc = (x + 0.5f) / (float)width;
    float y_ndc = (y + 0.5f) / (float)height;
    float x_camera = (2.0f * x_ndc - 1.0f) * aspect_ratio * tan_half_fov;
    float y_camera = (1.0f - 2.0f * y_ndc) * tan_half_fov;
    float3 ray_direction = (float3)(x_camera, y_camera, -1.0f);
    ray_direction = normalize(ray_direction);

    // Rotate the ray direction based on camera angle
    float cos_angle = cos(camera_angle);
    float sin_angle = sin(camera_angle);
    ray_direction = (float3)(cos_angle * ray_direction.x - sin_angle * ray_direction.z,
                             ray_direction.y,
                             sin_angle * ray_direction.x + cos_angle * ray_direction.z);

    // Create the ray origin
    float3 ray_origin = camera_origin.xyz;

    float hit_dist = -1.0f;
    float3 hit_position = (float3)(0.0f, 0.0f, 0.0f);
    float3 hit_normal = (float3)(0.0f, 0.0f, 0.0f);
    float2 hit_uv = (float2)(0.0f, 0.0f);

    const int MAX_PORTAL_DEPTH = 5;
    int portal_depth = 0;

    #ifdef DEBUG
    float3 debug_pixel_color = (float3)(0.0f, 0.0f, 0.0f);
    #endif

    while (portal_depth < MAX_PORTAL_DEPTH) {
        hit_dist = -1.0f;
         
        ray_trace_bvh(
            ray_origin,
            ray_direction,
            triangle_verticies_g,
            triangle_indicies_g,
            triangle_uvs_g,
            triangle_uv_indicies_g,
            bvh_structs_g,
            triangle_count,
            &hit_dist,
            &hit_position,
            &hit_normal,
            &hit_uv,
            bvh_debug_depth
            #ifdef DEBUG
            , &debug_pixel_color
            #endif
        );

        bool portal_hit = false;
        int portal_index = -1;
    
        // Check if we hit any portal
        for (int i = 0; i < portal_count; i++) {
            float portal_hit_dist = -1.0f;
            portal_t portal = portal_g[i];
    
            check_portal_intersection(
                ray_origin,
                ray_direction,
                portal,
                &portal_hit_dist
            );
    
            if (portal_hit_dist > 0.0f && (portal_hit_dist < hit_dist || hit_dist < 0.0f)) {
                hit_dist = portal_hit_dist;
                portal_index = i;
                portal_hit = true;
            }
    
        }

        if (bvh_debug_depth != 0 || !portal_hit || hit_dist < 0.0f) {
            break; // No portal hit or no intersection found or we are in debug mode
        }

        if (portal_index != -1) {
            // Transform the ray to be where it would leave the portal
            portal_t portal = portal_g[portal_index];
            ray_origin = ray_origin + (ray_direction * hit_dist); // Move ray origin to the portal hit position
            ray_origin = mat4_mul_vec4(portal.ray_transform, (float4)(ray_origin, 1.0f)).xyz;
            ray_direction = mat4_mul_vec4(portal.ray_transform, (float4)(ray_direction, 0.0f)).xyz;
            ray_direction = normalize(ray_direction);
            ray_origin = ray_origin + (ray_direction * 0.001f); // Move slightly away from the portal to avoid hitting it again
        }
        portal_depth++;
    }

    #ifdef DEBUG
    if (bvh_debug_depth != 0) {
        res_g[(y * width) + x] = (float4)(debug_pixel_color, 1.0f); // Set pixel color for debug
        return;
    }
    #endif

    // Shade
    float3 pixel_color = (float3)(0.0f, 0.0f, 0.0f);

    if (hit_dist >= 0.0f) {
        hit_uv.y = 1.0f - hit_uv.y;
        float4 color = read_imagef(world_texture, world_texture_sampler, hit_uv);

        float3 light_position = (float3)(-200.0f, 300.0f, 500.0f); // Example light position
        float3 light_direction = normalize(light_position - hit_position);
        float diffuse = max(dot(hit_normal, light_direction), 0.1f);
        pixel_color = (float3)(diffuse, diffuse, diffuse); // Simple
        pixel_color = diffuse * color.xyz; // color.xyz; // * COLORS[index % COLOR_COUNT]; // Use triangle_start as an index for color
    }

    // if(portal_depth == 0) {
    //     res_g[(x * height) + y] = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
    //     return;
    // } else if (portal_depth == 1) {
    //     res_g[(x * height) + y] = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
    //     return;
    // } else if (portal_depth == 2) {
    //     res_g[(x * height) + y] = (float4)(0.0f, 0.0f, 1.0f, 1.0f);
    //     return;
    // } else if (portal_depth == 3) {
    //     res_g[(x * height) + y] = (float4)(1.0f, 1.0f, 0.0f, 1.0f);
    //     return;
    // }

    // res_g[(x * height) + y] = (float4)(hit_uv.x, hit_uv.y, 0.0f, 1.0f); // Set pixel color
    // return;
    res_g[(y * width) + x] = (float4)(pixel_color, 1.0f); // Set pixel color

}

// __kernel void dlss(
//     __global float4* res_g
// )
// {
//     int x = get_global_id(0);
//     int y = get_global_id(1);

//     int width = get_global_size(0);
//     int height = get_global_size(1);

//     if (y % 2 == 0 || x % 2 == 0) {
//         float3 sum = (float3)(0.0f, 0.0f, 0.0f);
//         int count = 0;
//         for (int xx = x-1; xx <= x+1; xx++) {
//             for (int yy = y-1; yy <= y+1; yy++) {
//                 if (xx < 0 || xx >= width || yy < 0 || yy >= height) {
//                     continue; // Skip out of bounds pixels
//                 }
//                 if (res_g[(yy * width) + xx].w == 0.0f) {
//                     continue; // Skip pixels that are not set
//                 }
//                 float4 color = res_g[(yy * width) + xx];
//                 sum += color.xyz;
//                 count++;
//             }
//         }

//         if (count > 0) {
//             sum /= (float)count; // Average the color
//             res_g[(y * width) + x] = (float4)(sum, 1.0f); // Set pixel color
//         } else {
//             res_g[(y * width) + x] = (float4)(0.0f, 0.0f, 0.0f, 1.0f); // Set pixel to black if no neighbors
//         }
//     }
// }