#ifndef __GLOBAL_H
#define __GLOBAL_H

#include <iostream>
using namespace std;

#include <utilities.h>
#include <parameters.h> /* Parameter file reader definition */

extern parameters *p;   /* And the global parameter pointer */
extern int verbose;

enum {
  INPUT = 1,
  HIDDEN,
  OUTPUT,
};

#endif
