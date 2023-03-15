//
// Created by jzy99 on 2023/3/11.
//

#ifndef LEARNOPTIX_LAUCHPARAMS_H
#define LEARNOPTIX_LAUCHPARAMS_H

#include <cstdint>

struct vec2i{
    int x,y;
};

struct LaunchParams
{
    int       frameID { 0 };
    uint32_t *colorBuffer;
    vec2i     fbSize;
};

#endif //LEARNOPTIX_LAUCHPARAMS_H
