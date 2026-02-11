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
