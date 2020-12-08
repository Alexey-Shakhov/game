gcc -I./cglm/include -lglfw -lvulkan -lm \
    globals.h utils.h utils.c render.h render.c main.c alloc.h alloc.c \
    -o game
