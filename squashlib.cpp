#include <math.h>
#include <string>
#include "squashlib.h"

static float temperature;

/***********************************************
*                 Export table                 *
***********************************************/
static float (*function)          (float);
static float (*derrivitive)       (float);
static float (*error)             (float,float);
static float (*error_derrivitive) (float,float);

float squash_function( float v ) {
  return function(v);
}
float squash_derrivitive( float v ) {
  return derrivitive(v);
}
float squash_error( float a, float d ) {
  return error(a,d);
}
float squash_error_derrivitive( float a, float d ) {
  return error_derrivitive(a,d);
}

/***********************************************
*                 Error table                  *
***********************************************/
static float linear_error( float actual, float desired ) {
  return desired - actual;
}

static float linear_error_derrivitive( float actual, float desired ) {
  return 1.0f;
}

static float abs_error( float actual, float desired ) {
  return fabs(desired-actual);
}

static float abs_error_derrivitive( float actual, float desired ) {
  return 1.0f;
}

static float squared_error( float actual, float desired ) {
  return 0.5 * (desired-actual) * (desired-actual);
}

static float squared_error_derrivitive( float actual, float desired ) {
  return (actual-desired); // -(target-output)
}

/***********************************************
*               Function table                 *
***********************************************/
static float logistic( float value ) {
  return 1.0f/(1.0f+exp(-value));
}

static float logistic_derrivitive( float value ) {
  return logistic(value) * (1.0f - logistic(value));
}

static float sigmoid( float value ) {
  return 1.0f/(1.0f+exp(-2.0f*value/temperature));
}

static float sigmoid_derrivitive( float value ) {
  return (2.0f/temperature)*sigmoid(value)*(1.0f - sigmoid(value));
}

static float tanh( float value ) {
  return tanh(value);
}

static float tanh_derrivitive( float value ) {
  return 1.0f - tanh(value)*tanh(value);
}

static float identity( float value ) {
  return value;
}

static float identity_derrivitive( float value ) {
  return 1.0f;
}

void initialize_squashlib( std::string which, std::string err, float t ) {

  // Set up the selected squashing function
  if ( which == "SIGMOID" ) {
    function    = &sigmoid;
    derrivitive = &sigmoid_derrivitive;
  } 
  else if ( which == "IDENTITY" ) {
    function    = &identity;
    derrivitive = &identity_derrivitive;
  } 
  else if ( which == "TANH" ) {
    function    = &tanh;
    derrivitive = &tanh_derrivitive;
  }
  else if ( which == "LOGISTIC" ) {
    function    = &logistic;
    derrivitive = &logistic_derrivitive;
  }
  else {
    printf("Unknown weighting function!\n");
    exit(1);
  }

  if ( err == "LINEAR" ) {
    error             = &linear_error;
    error_derrivitive = &linear_error_derrivitive;
  }
  else if ( err == "ABS" ) {
    error             = &abs_error;
    error_derrivitive = &abs_error_derrivitive;
  }
  else if ( err == "SQUARED" ) {
    error             = &squared_error;
    error_derrivitive = &squared_error_derrivitive;
  }
  else {
    printf("Unknown error function\n");
    exit(1);
  }

  temperature = t;
  if ( !temperature )
    temperature = 1.0f;

  return;
}
