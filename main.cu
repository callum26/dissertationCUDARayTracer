// 3d vector structure
// models 3d vector with basic operations
struct Vec3
{
    float x, y, z;

    __device__ __host__ Vec3() : x(0), y(0), z(0) {}
    __device__ __host__ Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    __device__ __host__ Vec3 operator+(const Vec3 &o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    __device__ __host__ Vec3 operator-(const Vec3 &o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    __device__ __host__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    // dot product
    __device__ __host__ float dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }

    // vector length
    __device__ __host__ float length() const { return sqrtf(x * x + y * y + z * z); }

    // normalize vector to unit length
    __device__ __host__ Vec3 normalize() const
    {
        float L = length();
        return L > 0 ? Vec3(x / L, y / L, z / L) : Vec3();
    }
};

// sphere structure
// sphere with center radius and color
struct Sphere
{
    Vec3 c;
    float r;
    Vec3 color;
    __device__ __host__ Sphere(const Vec3 &C, float R, const Vec3 &col) : c(C), r(R), color(col) {}
};

// ray structure
// represents a ray with an origin and direction
struct Ray
{
    Vec3 o, d;

    
    __device__ __host__ Ray(const Vec3 &O, const Vec3 &D) : o(O), d(D) {}
    // point along ray at distance t
    __device__ __host__ Vec3 at(float t) const { return o + d * t; }
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
__device__ __host__ bool raySphereIntersect(const Ray &ray, const Sphere &s, float &t)
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
    float sq = sqrtf(disc);
    float t0 = (-b - sq) / (2 * a); // near intersection, entry point
    float t1 = (-b + sq) / (2 * a); // far intersection, exit

    // returns closest positive intersection to the camera first
    // if t0 > 0, ray hits sphere from outside, use entry point
    // if t0 < 0 but t1 > 0, camera must be inside sphere so it uses exit point
    t = (t0 > 0) ? t0 : t1;
    return t > 0;
}




__global__ void renderPixel(unsigned char* bits, Sphere* sphere, int numSpheres, Vec3 cam, Vec3 light, Vec3 groundN, float groundY, Vec3 groundC, int w, int h, float vw, float vh)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= h || j >= w) return; // check bounds of threads

        // ray trace each pixel

            // calculate normalised coords
            float u = (float)j / (w - 1);
            float v = (float)i / (h - 1);

            // convert to viewport coords
            float x = (u - 0.5f) * vw;
            float y = (0.5f - v) * vh;

            // create ray through pixel
            Ray r(cam, Vec3(x, y, -1).normalize());

            // test sphere intsect
            const Sphere *hitSphere = nullptr;
            float tS = 1e50f;
            for (int k = 0; k < numSpheres; ++k)
            {
                float tt;
                if (raySphereIntersect(r, sphere[k], tt) && tt < tS)
                {
                    tS = tt;
                    hitSphere = &sphere[k];
                }
            }

            // test ray plane intersection for ground plane
            // plane equation, all points P where (P - pointOnPlane) ⋅ normal = 0
            // for horiz ground y = groundY, normally = (0, 1, 0)
            // subing ray equation P = o + t * d -> plane equation becomes (origin.y + t * direction.y - groundY) = 0
            // then t = (groundY - origin.y) / direction.y
            bool hitP = false;
            float tP = 1e50f;

            // test if not parallel to plane 
            // small num to avoid division by near zero
            if (fabsf(r.d.y) > 1e-6f)
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
            int writeRow = (h - 1 - i);
            int pix = (writeRow * w + j) * 4;

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
                float diff = fmaxf(0.0f, N.dot(L));

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
                // higher exponent -> smaller, sharper highlight = shinier surface
                // max with 0 to ignore reflections away from viewer
                float sp = powf(fmaxf(0.0f, V.dot(R)), 32.0f);

                // combine lighting
                float rc = hitSphere->color.x * (ambient + diffuse * diff) + specular * sp;
                float gc = hitSphere->color.y * (ambient + diffuse * diff) + specular * sp;
                float bc = hitSphere->color.z * (ambient + diffuse * diff) + specular * sp;

                // write pixel clamped to valid range
                bits[pix + 0] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, rc)) * 255.0f);
                bits[pix + 1] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, gc)) * 255.0f);
                bits[pix + 2] = (unsigned char)(fminf(1.0f, fmaxf(0.0f, bc)) * 255.0f);
                bits[pix + 3] = 255;
            }
            else if (hitP)
            {
                // ground plane hit
                Vec3 P = r.at(tP);
                Vec3 L = (light - P).normalize();

                // simple diffuse lighting
                float diff = fmaxf(0.0f, groundN.dot(L));
                float inten = fminf(1.0f, fmaxf(0.0f, 0.2f + 0.7f * diff));

                // write ground color
                bits[pix + 0] = (unsigned char)(groundC.x * inten * 255.0f);
                bits[pix + 1] = (unsigned char)(groundC.y * inten * 255.0f);
                bits[pix + 2] = (unsigned char)(groundC.z * inten * 255.0f);
                bits[pix + 3] = 255;
            }
            else
            {
                // background miss
                if (i < h / 2)
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
    
        
main(){

}
    