#ifndef __LAYER_H
#define __LAYER_H

#include "node.h"

class layer {

 public:

  layer (void) {return;}
  layer(int, int, int=0,int=0);
  ~layer(void);

  int     get_number_of_weights ( void )      { return number_of_inputs * number_of_nodes; }
  int     get_number_of_nodes   ( void )      { return number_of_nodes; }
  float * get_norms             ( void )      { return normalization; }
  float * get_offsets           ( void )      { return input_offset; }

  layer * get_previous          ( void )      { return previous; }
  layer * get_next              ( void )      { return next; }

  void    set_previous          ( layer * p ) { previous = p; return; }
  void    set_next              ( layer * n ) { next = n; return ; }

  void    set_weights           ( int=0 );
  void    adjust_weights        ( void );
  float * get_weights           ( void );
  void    dump_weights          ( void );

  void    get_node_outputs      ( float * );
  void    normalize_input       ( float *, float * );

  float * get_node_errors       ( void );

  float * input                 ( float * );
  void    propagate_errors      ( void );

  void    set_output_targets    ( float * );

 private:

  layer * previous;
  layer * next;

  int     layer_number;
  int     number_of_inputs;
  int     layer_type;
  int     number_of_nodes;
  int     number_of_output_nodes;

  bool    bias;

  float * weights;
  float * back_errors;
  float * result;
  float * output;
  float * normalization;
  float * input_offset;
  
  node ** nodes;

};

#endif
