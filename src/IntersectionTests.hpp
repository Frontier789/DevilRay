HD std::optional<Intersection> getIntersection(const Ray &ray, const Square &square)
{
    // <p + v*t - o, n> = 0
    // <po, n> + <v,n>*t = 0


    const auto vn = dot(ray.v, square.n);
    if (std::abs(vn) < std::numeric_limits<float>::epsilon()) return std::nullopt;

    const auto t = -dot(ray.p - square.p, square.n) / vn;

    // std::cout << ray.p.x << "," << ray.p.y << "," << ray.p.z << " -> "  << ray.v.x << "," << ray.v.y << "," << ray.v.z << " -> t=" << t << std::endl;
    if (t < std::numeric_limits<float>::epsilon()) return std::nullopt;

    const auto p = ray.p + ray.v * t;
    const auto s = square.size / 2;

    const auto dx = dot(square.right, p - square.p);
    if (std::abs(dx) > s) return std::nullopt;

    const auto up = square.right.cross(square.n);
    const auto dy = dot(up, p - square.p);
    if (std::abs(dy) > s) return std::nullopt;

    const auto uv = Vec2{
        dx / s / 2 + 0.5f,
        dy / s / 2 + 0.5f,
    };

    const auto n = vn < 0 ? square.n : square.n * -1;

    return Intersection{
        .t = t,
        .p = p,
        .uv = uv,
        .n = n,
        .mat = square.mat,
    };
}

HD std::optional<Intersection> getIntersection(const Ray &ray, const Sphere &sphere)
{
    // |p + v*t - o| = r
    // <po + v*t, po + v*t> = r^2
    // |po|^2 + 2<po, v> * t + |v|^2 * t^2 = r^2

    const auto o = ray.p - sphere.center;

    const auto a = dot(ray.v, ray.v);
    const auto b = 2*dot(o, ray.v);
    const auto c = dot(o,o) - sphere.radius * sphere.radius;

    const auto D = b*b - 4*a*c;
    if (D < 0) return std::nullopt;

    const auto d = std::sqrt(D);

    const float q = -0.5f * ((b > 0) ? (b + d) : (b - d));
    const float t1 = q / a;
    const float t2 = c / q;

    const auto tmin = std::min(t1, t2);
    const auto tmax = std::max(t1, t2);

    if (tmax < 0) return std::nullopt;

    const auto t = tmin < 0 ? tmax : tmin;
    const auto p = ray.p + ray.v * t;

    const auto pl = p - sphere.center;

    const auto uv = Vec2f{
        .x = std::atan2(pl.z, pl.x) / std::numbers::pi_v<float> / 2 + 0.5f,
        .y = std::acos(pl.y / sphere.radius) / std::numbers::pi_v<float>,
    };

    auto n = (p - sphere.center).normalized();
    if (c < -1e-8) n = n * -1;

    return Intersection{
        .t = t,
        .p = p,
        .uv = uv,
        .n = n,
        .mat = sphere.mat,
    };
}