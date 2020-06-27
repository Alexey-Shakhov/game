#version 450
#extension GL_ARB_separate_shader_objects : enable

vec4 qmul(vec4 q1, vec4 q0);

layout(binding = 0) uniform UniformBuffer {
    vec3 model_translation;
    vec4 model_rotation;
    vec3 model_scale;
    vec3 view_translation;
    vec4 view_rotation;
    vec3 view_scale;
    mat4 projection;
} mvp;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 outUV;

out gl_PerVertex {
    vec4 gl_Position;
};

vec4 qmul(vec4 q1, vec4 q0) {
    return vec4(
        q1.x * q0.x - dot(q1.yzw, q0.yzw),
        q1.x * q0.yzw + q0.x * q1.yzw + cross(q1.yzw, q0.yzw)
    );
}

vec4 qinverse(vec4 q) {
    return vec4(q.x, -q.yzw);
}

vec3 rotate(vec4 q, vec3 v) {
    return qmul(q, qmul(vec4(0.0, v), qinverse(q))).yzw;
}

void main() {
    vec3 position;

    position = position + mvp.model_translation;
    position = rotate(mvp.model_rotation, position);
    position = mvp.model_scale * position;

    position = position + mvp.view_translation;
    position = rotate(mvp.view_rotation, position);
    position = mvp.view_scale * position;

    gl_Position = mvp.projection * vec4(position, 1.0);

    outUV = uv;
}
