gcc -I./cglm/include -lglfw -lvulkan -lm \
    utils.h utils.c platform.h linwin.c alloc.h alloc.c \
    -o game
