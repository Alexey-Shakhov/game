#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 color;
layout(location = 1) flat in uint code;

layout(location = 0) out vec4 out_color;
layout(location = 1) out uint out_code;

void main() {
    out_color = vec4(color, 1.0);    
    out_code = code;
}
