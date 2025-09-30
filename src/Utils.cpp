#include "Utils.hpp"


Timer::Timer() : start_time(std::chrono::steady_clock::now()) {}

float Timer::elapsed_seconds() const
{
    const auto delta = std::chrono::steady_clock::now() - start_time;
    const auto mics = std::chrono::duration_cast<std::chrono::microseconds>(delta);
    return mics.count() / 1e6f;
}

Vec4 checkerPattern(
    const Vec2f &uv, 
    const int checker_count, 
    const Vec4 dark, 
    const Vec4 bright
){
    const auto checker_x = int(uv.x * checker_count) % 2;
    const auto checker_y = int(uv.y * checker_count) % 2;

    const float checker = checker_x ^ checker_y;

    return bright * checker + dark * (1-checker);
}


int Random::id_counter = 0;

Random RandomPool::borrowRandom()
{
	std::lock_guard<std::mutex> guard(m_mutex);

	if (m_randoms.empty()) m_randoms.push_back(Random{});
	Random r = std::move(m_randoms.back());
	m_randoms.pop_back();
	
	return r;
}

void RandomPool::returnRandom(Random r)
{
	std::lock_guard<std::mutex> guard(m_mutex);
	
	m_randoms.emplace_back(std::move(r));	
}

RandomPool &RandomPool::singleton()
{
    static RandomPool pool;

    return pool;
}


std::ostream &operator<<(std::ostream &os, const ColorRGBA8 &color) {
    os << "(" << static_cast<int>(color.r) << ", " 
       << static_cast<int>(color.g) << ", " 
       << static_cast<int>(color.b) << ", " 
       << static_cast<int>(color.a) << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Vec4 &vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Vec3 &vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    os << "Ray(p=" << ray.p << ", v=" << ray.v << ")";
    return os;
}