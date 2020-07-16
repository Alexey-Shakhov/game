gcc -I./cglm/include -lglfw -lvulkan -lm \
    utils.h utils.c render.h render.c main.c alloc.h alloc.c \
    -o game
