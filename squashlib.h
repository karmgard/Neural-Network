#ifndef __SQUASHLIB_H
#define __SQUASHLIB_H
void    initialize_squashlib     ( std::string, std::string, float=1.0f );
float   squash_function          ( float );
float   squash_derrivitive       ( float );
float   squash_error             ( float,float );
float   squash_error_derrivitive ( float,float );
#endif
