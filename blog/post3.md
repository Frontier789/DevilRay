# Materials

Previously I started working on my very own ray tracer in C++. I got to render basic shapes and put a checker texture on them.

This time I want to introduce a new concept into the code, materials.

## Debug color

Each object looks the same in my render right now, it would be nice to see where one ends and another begins.

To achieve that I added a material class:
```C++
struct Material
{
    Vec4 debug_color;
};
```

Each object will now have a material as well:
```C++
struct Sphere
{
    ...
    Material *mat;
};
```

I use a raw pointer for the material, meaning its non-owning. So before I populate the objects array I have to create materials, and store them.
```C++
Material blue;
...
blue = Material{
    .debug_color = Vec4(.7,.8,.9,0);
};
```

And then when creating objects:
```C++
objects.emplace_back(Sphere{
    ...
    .mat = &blue,
});
```

Finally the intersection class needs to store the material as well. This is populated when the ray hits an object.
```C++
struct Intersection
{
    ...
    Material *mat;
};
```

With these changes I get a nice colored room:
![](images/checker_colored.png)

## Future compatibility

Right now the material class only includes the debug color, but later I want to use it to describe more properties of the object. 

In the end I would like to be able to create different kinds of materials and combine them, for that I will need a unified approach to all operations done on materials and for now I am not sure what these operations will be. \
Sampling an outgoing ray, given the inbound ray is the most important I can think of now.

So for now I decided to roll with a single material class that I will add more members to.
