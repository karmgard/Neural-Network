#ifndef __NODE_H
#define __NODE_H

#include <math.h>
#include <stdlib.h>
#include <utility>
using namespace std;

class node {

 public:

  node(void) {return;}
  node(int,int,int=0);
  node( const node & );
  ~node(void);

  float   input                ( float * );
  void    adjust_weights       ( void );
  void    dump_weights         ( void );
  int     set_weights          ( int );
  float * get_weights          ( void )     {return weights;}
  void    calculate_error      ( float );

  int     get_type             ( void )     {return node_type;}
  void    set_type             ( int type ) {node_type=type; return;}

  float   get_node_bias        ( void )     { return bias; }
  void    set_node_bias        ( float b )  { bias=b; return; }

  float   get_node_target      ( void )     { return target; }
  void    set_node_target      ( float t )  { target=t; return; }

  float   get_node_error       ( void )     { return node_error; }

  friend void swap( node & f, node & s ) {
    using std::swap;
    swap( f.number_of_inputs, s.number_of_inputs );
    swap( f.node_number,      s.node_number      );
    swap( f.weights,          s.weights          );
    return;
  }

  node & operator = (const node & rhs){node n(rhs);swap(*this,n);return *this;}
  node & operator = (const node * rhs){
    node *n=(node *)rhs;swap(*this,*n);return *this;
  }

 private:

  int     number_of_inputs;
  int     node_number;
  int     node_type;
  float   target;
  float   learning_rate;
  float   node_error;
  float   weighted_sum;
  float   output;
  float   bias;

  float * weights;
  float * last_input;

};

#endif
