#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in uint code;

layout(location = 0) out vec3 out_color;
layout(location = 1) out uint out_code;

layout(binding=0) uniform Uniform {
    mat4 view_proj;
} uni;

out gl_PerVertex {
    vec4 gl_Position;
    float gl_PointSize;
};

void main() {
    gl_Position = uni.view_proj * vec4(position, 1.0);
    gl_PointSize = 10.0;
    out_color = color;
    out_code = code;
}
