#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex_coord;

layout(location = 0) out vec2 out_tex_coord;
layout(location = 1) out vec4 out_world_pos;

layout(binding=0) uniform Uniform {
    mat4 view_proj;
} uni;

layout(push_constant) uniform PushConsts {
    mat4 model;
} push_consts;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = uni.view_proj * push_consts.model * vec4(position, 1.0);
    out_tex_coord = tex_coord;
    out_world_pos = push_consts.model * vec4(position, 1.0);
}
