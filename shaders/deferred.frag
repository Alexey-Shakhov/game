#version 450

layout(set=2, binding=0) uniform sampler2D position;
layout(set=2, binding=1) uniform sampler2D normal;
layout(set=2, binding=2) uniform sampler2D albedo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 out_color;

void main()
{
    vec3 pos = texture(position, inUV).rgb;
    vec3 normal = texture(normal, inUV).rgb;
    vec4 albedo = texture(albedo, inUV);

    #define ambient 0.5
    out_color = albedo * ambient;
}
