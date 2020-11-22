#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform sampler2D tex_sampler;

layout(location = 0) in vec3 color;
layout(location = 1) in vec2 tex_coord;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(tex_sampler, tex_coord);
}
