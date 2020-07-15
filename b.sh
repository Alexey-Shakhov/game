gcc -I./cglm/include -lglfw -lvulkan -lm \
    utils.h utils.c vkrend.h vkrend.c alloc.h alloc.c main.c \
    -o game
