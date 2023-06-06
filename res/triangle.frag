#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec4 fragColor;

void main() {
	fragColor = vec4(position, 1.0);
}
