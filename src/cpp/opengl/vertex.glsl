attribute vec4 position;
void main()
{
   gl_Position = vec4(position.xyz, 1.0);
}