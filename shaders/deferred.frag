#version 450

layout(set=2, binding=0) uniform sampler2D position;
layout(set=2, binding=1) uniform sampler2D normal;
layout(set=2, binding=2) uniform sampler2D albedo;

layout(binding=1) uniform DeferredUbo {
    vec3 view_pos;
    uint light_count;
} ubo;

struct Light {
    vec3 position;
    vec3 color;
};

layout(binding=2) buffer LightsSB
{
    Light lights[];  
};

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 out_color;

void main()
{
    vec3 pos = texture(position, inUV).rgb;
    vec3 normal = texture(normal, inUV).rgb;
    vec4 albedo = texture(albedo, inUV);

    #define ambient 0.3
    vec3 color = albedo.rgb * ambient;

    for (int i=0; i < ubo.light_count; i++) {
        Light light = lights[i];

        vec3 frag_light_vec = light.position - pos;
        frag_light_vec = normalize(frag_light_vec);
        vec3 frag_view_dir = normalize(ubo.view_pos - pos);

        float dot_nl = max(0.0, dot(normal, frag_light_vec));
        vec3 diff = light.color * albedo.rgb * dot_nl;

        color += diff;
    }

    out_color = vec4(color, 1.0);
}
