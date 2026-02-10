// imports
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>

// window dimensions
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

// 3d vector structure
// models 3d vector with basic operations
struct Vec3
{
    float x, y, z;

    // constructors f
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    // operators
    Vec3 operator+(const Vec3 &o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3 &o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    // dot product
    float dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }

    // vector length
    float length() const { return std::sqrt(x * x + y * y + z * z); }

    // normalize vector to unit length
    Vec3 normalize() const
    {
        float L = length();
        return L > 0 ? Vec3(x / L, y / L, z / L) : Vec3();
    }
};

// ray structure
// represents a ray with an origin and direction
struct Ray
{
    Vec3 o, d;
    Ray(const Vec3 &O, const Vec3 &D) : o(O), d(D) {}

    // point along ray at distance t
    Vec3 at(float t) const { return o + d * t; }
};

// sphere structure
// sphere with center radius and color
struct Sphere
{
    Vec3 c;
    float r;
    Vec3 color;
    Sphere(const Vec3 &C, float R, const Vec3 &col) : c(C), r(R), color(col) {}
};

// ray sphere intersection function
// returns true if ray hits sphere and sets t to distance
// uses quadratic formula to solve for intsec points
// ray equation is P(t) = o + t * d  where o = origin of ray, d = direction the ray points to and t >= 0
// vector definition of a sphere is: |P - center|^2 = radius^2
// subing it in gives: at^2 + bt + c = 0 where:
//   a = d·d (direction dot direction)
//   b = 2 * (origin - center)·d
//   c = |origin - center|^2 - radius^2
bool raySphereIntersect(const Ray &ray, const Sphere &s, float &t)
{
    // vector from sphere center to ray origin
    Vec3 oc = ray.o - s.c;

    // quadratic coefficients for at^2 + bt + c = 0
    float a = ray.d.dot(ray.d);       // always 1 if direction is normalised
    float b = 2.0f * oc.dot(ray.d);   // projection of oc onto ray direction
    float c = oc.dot(oc) - s.r * s.r; // squared distance minus squared radius for hit test

    // just like regular quadratics discriminant of this equation determines number of intersections:
    // < 0 no intersection, ray misses
    // = 0 one intersection, ray is tangent to spehere
    // > 0 two intersections, ray passes through sphere
    float disc = b * b - 4 * a * c; // discriminant

    // no intersection, ray misses sphere entirely
    if (disc < 0)
        return false;

    // quadratic formula to calculate both of intersection distances
    // t = (-b ± sqrt(discriminant)) / 2a
    float sq = std::sqrt(disc);
    float t0 = (-b - sq) / (2 * a); // near intersection, entry point
    float t1 = (-b + sq) / (2 * a); // far intersection, exit

    // returns closest positive intersection to the camera first
    // if t0 > 0, ray hits sphere from outside, use entry point
    // if t0 < 0 but t1 > 0, camera must be inside sphere so it uses exit point
    t = (t0 > 0) ? t0 : t1;
    return t > 0;
}

// read shader file
// loads shader source code from file path
std::string readShader(const char *path)
{
    std::ifstream in(path);
    if (!in)
        return std::string();
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// compile shader
// compiles shader source and returns shader id
unsigned int compileShader(unsigned int type, const char *src)
{
    unsigned int s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    // check compilation status
    int ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char b[512];
        glGetShaderInfoLog(s, 512, NULL, b);
        std::cerr << "shader compile error " << b << "\n";
    }
    return s;
}

// create shader program
// links vertex and fragment shaders into program
unsigned int createProgram(const char *vpath, const char *fpath)
{
    // load shader sources
    std::string vs = readShader(vpath);
    std::string fs = readShader(fpath);
    if (vs.empty() || fs.empty())
    {
        std::cerr << "missing shader files\n";
        return 0;
    }

    // compile shaders
    unsigned int vsId = compileShader(GL_VERTEX_SHADER, vs.c_str());
    unsigned int fsId = compileShader(GL_FRAGMENT_SHADER, fs.c_str());

    // link program
    unsigned int p = glCreateProgram();
    glAttachShader(p, vsId);
    glAttachShader(p, fsId);
    glLinkProgram(p);

    // check link status
    int ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char b[512];
        glGetProgramInfoLog(p, 512, NULL, b);
        std::cerr << "shader link error " << b << "\n";
    }

    // cleanup shaders
    glDeleteShader(vsId);
    glDeleteShader(fsId);
    return p;
}

int main()
{
    // init glfw
    if (!glfwInit())
        return -1;

    // set opengl version 3.3 core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

// macos compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // create window
    GLFWwindow *win = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Ray Tracer", NULL, NULL);
    if (!win)
    {
        glfwTerminate();
        return -1;
    }

    // make context current
    glfwMakeContextCurrent(win);

    // init glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "glad init failed\n";
        return -1;
    }

    // create shader program
    unsigned int prog = createProgram("../shaders/vertex.glsl", "../shaders/fragment.glsl");

    // fullscreen quad vertices
    // position xy and texture coords uv
    float verts[] = {
        1, 1, 1, 1,   // top right
        1, -1, 1, 0,  // bottom right
        -1, -1, 0, 0, // bottom left
        -1, 1, 0, 1   // top left
    };

    // triangle indices for quad
    unsigned int idx[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    // create vertex array object vertex buffer object element buffer object
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    // upload index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // create texture for raytraced image
    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // allocate pixel buffer
    int W = SCR_WIDTH;
    int H = SCR_HEIGHT;
    std::vector<unsigned char> bits(W * H * 4);

    // setups camera at world origin
    Vec3 cam(0, 0, 0);
    float fov = 90.0f;
    float aspect = (float)W / (float)H;

    // viewport dims
    float vh = 2.0f;
    float vw = aspect * vh;

    // create scene with single sphere
    std::vector<Sphere> scene;
    scene.emplace_back(Vec3(0, 0, -5), 1.5f, Vec3(0.2f, 0.5f, 1.0f));

    // light position
    Vec3 light(3, 3, 2);

    // ground plane settings
    Vec3 groundN(0, 1, 0);
    float groundY = -2.0f;
    Vec3 groundC(0.6f, 0.6f, 0.6f);

    // ray trace each pixel
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            // calculate normalised coords
            float u = (float)j / (W - 1);
            float v = (float)i / (H - 1);

            // convert to viewport coords
            float x = (u - 0.5f) * vw;
            float y = (0.5f - v) * vh;

            // create ray through pixel
            Ray r(cam, Vec3(x, y, -1).normalize());

            // test sphere intsect
            const Sphere *hitSphere = nullptr;
            float tS = std::numeric_limits<float>::infinity();
            for (auto &s : scene)
            {
                float tt;
                if (raySphereIntersect(r, s, tt) && tt < tS)
                {
                    tS = tt;
                    hitSphere = &s;
                }
            }

            // test ray plane intersection for ground plane
            // plane equation, all points P where (P - pointOnPlane) ⋅ normal = 0
            // for horiz ground y = groundY, normally = (0, 1, 0)
            // subing ray equation P = o + t * d -> plane equation becomes (origin.y + t * direction.y - groundY) = 0
            // then t = (groundY - origin.y) / direction.y
            bool hitP = false;
            float tP = std::numeric_limits<float>::infinity();

            // only test if ray is NOT parallel to plane (direction.y != 0)
            // use small epsilon to avoid division by near-zero
            if (std::fabs(r.d.y) > 1e-6f)
            {
                // calculate intersection distance
                float tc = (groundY - r.o.y) / r.d.y;

                // only counts hits in front of camera (t > 0) otherwise there would be too many hits behind camera
                if (tc > 0)
                {
                    tP = tc;
                    hitP = true;
                }
            }

            // calculate pixel index flipped for opengl
            int writeRow = (H - 1 - i);
            int pix = (writeRow * W + j) * 4;

            // shade closest hit
            if (hitSphere && (!hitP || tS < tP))
            {
                // sphere hit
                Vec3 P = r.at(tS);
                Vec3 N = (P - hitSphere->c).normalize();
                Vec3 L = (light - P).normalize();

                // phong shading model
                // color = ambient + diffuse + specular
                // helps to approx. how light interacts with surfaces more realistically

                // diffuse factor is for how directly the surface faces the light
                // N · L = cos(angle), cos ranging from 1 (facing light) to -1 (facing away)
                // max with 0 to ignore surfaces facing away from light
                float diff = std::max(0.0f, N.dot(L));

                // lighting coefficients
                float ambient = 0.2f;  // constant base illumination, indirect light
                float diffuse = 0.7f;  // matte surface reflection strength
                float specular = 0.3f; // shiny highlight strength

                // specular highlight using phong reflection model
                // V = view direction (from hit point toward camera)
                Vec3 V = (cam - P).normalize();

                // R = reflection of light vector around surface normal
                // R = L - 2(L · N)N reflects L across plane perpendicular to N
                //
                Vec3 R = L - N * (2.0f * L.dot(N));

                // specular intensity:
                // (V·R)^shiniess
                // higher exponent (32) = smaller, sharper highlight = shinier surface
                // max with 0 to ignore reflections away from viewer
                float sp = std::pow(std::max(0.0f, V.dot(R)), 32.0f);

                // combine lighting
                float rc = hitSphere->color.x * (ambient + diffuse * diff) + specular * sp;
                float gc = hitSphere->color.y * (ambient + diffuse * diff) + specular * sp;
                float bc = hitSphere->color.z * (ambient + diffuse * diff) + specular * sp;

                // write pixel clamped to valid range
                bits[pix + 0] = (unsigned char)(std::min(1.0f, std::max(0.0f, rc)) * 255.0f);
                bits[pix + 1] = (unsigned char)(std::min(1.0f, std::max(0.0f, gc)) * 255.0f);
                bits[pix + 2] = (unsigned char)(std::min(1.0f, std::max(0.0f, bc)) * 255.0f);
                bits[pix + 3] = 255;
            }
            else if (hitP)
            {
                // ground plane hit
                Vec3 P = r.at(tP);
                Vec3 L = (light - P).normalize();

                // simple diffuse lighting
                float diff = std::max(0.0f, groundN.dot(L));
                float inten = std::min(1.0f, std::max(0.0f, 0.2f + 0.7f * diff));

                // write ground color
                bits[pix + 0] = (unsigned char)(groundC.x * inten * 255.0f);
                bits[pix + 1] = (unsigned char)(groundC.y * inten * 255.0f);
                bits[pix + 2] = (unsigned char)(groundC.z * inten * 255.0f);
                bits[pix + 3] = 255;
            }
            else
            {
                // background miss
                if (i < H / 2)
                {
                    // sky color
                    bits[pix + 0] = 0;
                    bits[pix + 1] = 0;
                    bits[pix + 2] = 0;
                    bits[pix + 3] = 255;
                }
                else
                {
                    // ground color
                    bits[pix + 0] = 255;
                    bits[pix + 1] = 0;
                    bits[pix + 2] = 0;
                    bits[pix + 3] = 255;
                }
            }
        }
    }

    // upload pixel data to texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, bits.data());

    // main render loop
    while (!glfwWindowShouldClose(win))
    {
        // escape key to close
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, true);

        // clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw textured quad
        glUseProgram(prog);
        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // swap buffers and poll events
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // cleanup
    glfwTerminate();
    return 0;
}
