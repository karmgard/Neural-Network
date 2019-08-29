#ifndef __NETWORK_H
#define __NETWORK_H

#include "layer.h"
#include <fstream>
#include <string>

class network {
 public: 
  network(void);
  ~network(void);

  float * input           ( float [] );
  void    dump_weights    ( void );
  void    set_weights     ( void );
  int     normalize       ( std::string );
  int     train_network   ( void );
  void    save_network    ( std::string );
  float   train           ( float [], float [] );

  int getNumberOfInputs   ( void)  {return number_of_inputs;}
  int getNumberOfOutputs  ( void)  {return number_of_outputs;}
  int getNumberOfLayers   ( void)  {return number_of_layers;}
  int getNodesPerLayer    ( int l) {
    return ( l>= 0 && l < number_of_layers ) ? topology[l] : -1;
  }

 private:

  int number_of_layers;
  int number_of_inputs;
  int number_of_outputs;
  int   * topology;

  float * result;

  layer * first;
  layer * last;

  void    parse_topology  (void);
};

#endif
