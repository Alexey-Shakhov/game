#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 tex_coord;
layout(location = 1) in vec4 in_world_pos;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_albedo;
layout(location = 3) out uint out_code;

void main() {
    out_position = in_world_pos;    
    out_normal = vec4(normalize(in_normal), 1.0);
    out_albedo = texture(tex_sampler, tex_coord);
    out_code = 120;
}
