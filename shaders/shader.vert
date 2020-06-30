#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 out_color;

layout(binding=0) uniform Uniform {
    mat4 projection;
} uni;

layout(push_constant) uniform PushConsts {
    mat4 model;
    mat4 view;
} push_consts;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = uni.projection * push_consts.view *
        push_consts.model * vec4(position, 1.0);
    out_color = vec3(1.0, 1.0, 1.0);
}
