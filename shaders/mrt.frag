#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 tex_coord;
layout(location = 1) in vec4 in_world_pos;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_albedo;

void main() {
    out_position = in_world_pos;    
    out_normal = vec4(0.5);
    out_albedo = texture(tex_sampler, tex_coord);
}
