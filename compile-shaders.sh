# compiles all the shaders

glslangValidator -H -V -o src/scores.spv src/scores.comp
glslangValidator -H -V -o src/scores2.spv src/scores2.comp
glslangValidator -H -V -o src/gradient.spv src/gradient.comp
glslangValidator -H -V -o src/gradient2.spv src/gradient2.comp
glslangValidator -H -V -o src/loss.spv src/loss.comp
