#version 330 core

in vec4 color;
out vec4 out_color;

uniform bool renderPointsAsSpheres;
uniform bool shadePointSpheres;

void main()
{
  if(renderPointsAsSpheres)
  {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if(r2 > 1.0) discard;

    if(!shadePointSpheres)
    {
      out_color = color;
      return;
    }

    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 normal = normalize(vec3(p, z));
    vec3 light = normalize(vec3(-0.35, 0.45, 0.82));
    float diffuse = max(dot(normal, light), 0.0);
    float shade = 0.45 + 0.55 * diffuse;
    out_color = vec4(color.rgb * shade, color.a);
    return;
  }

  out_color = color;
}
