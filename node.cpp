#include "global.h"
#include "node.h"
#include "squashlib.h"

#include <math.h>

node::node(int ni, int nn, int type) {
  number_of_inputs = ni;
  node_number      = nn;
  node_type        = type;

  // Initialize the arrays
  last_input = new float [number_of_inputs];
  weights    = new float [number_of_inputs];
  for ( int i=0; i<number_of_inputs; i++ ) {
    weights[i]    = randf();
    last_input[i] = 0.0f;
  }

  // Initialize all the class variables
  weighted_sum  = 0.0f;
  node_error    = 0.0f;
  target        = 0.0f;
  output        = 0.0f;
  bias          = 0.0f;

  // And read in the parameters that affect our calculations
  learning_rate = p->keyExists("LEARNING_RATE") ? p->getFloat("LEARNING_RATE") : 0.1;

  if ( verbose )
    cout << "Configed node " << node_number << " to receive " << number_of_inputs << " inputs\n";

  return;
}

node::node( const node & n ) {
  this->number_of_inputs = n.number_of_inputs;
  this->node_number      = n.node_number;
  this->bias             = n.bias;

  if ( !weights )
      weights = new float [number_of_inputs];

  for ( int i=0; i<number_of_inputs; i++ ) {
    this->weights[i]     = n.weights[i];
    this->last_input[i]  = n.last_input[i];
  }

  return;
}

node::~node(void) {
  if ( weights )
    delete [] weights;
  if ( last_input )
    delete [] last_input;

  return;
}

float node::input( float * input_array ) {
  if ( verbose > 1 ) {
    cout << "Node " << node_number << " got (";
    for ( int i=0; i<number_of_inputs; i++ )
      cout << input_array[i] << ", ";
    cout << "\b\b)\n";
  }

  weighted_sum = 0.0f;
  for ( int i=0; i<number_of_inputs; i++ ) {
    weighted_sum += weights[i] * input_array[i];    
    last_input[i] = input_array[i];
  }
  weighted_sum += bias;

  if ( verbose > 1 )
    cout << "Sending squasher " << weighted_sum << "...";

  output = squash_function(weighted_sum);

  if ( verbose > 1 )
    cout << "Node " << node_number << " is returning " << output << "\n";

  return output;
}

void node::calculate_error( float propagated_error ) {

  if ( node_type == OUTPUT )
    node_error = squash_error_derrivitive(output, target) * squash_derrivitive(weighted_sum);

  else {
    node_error = propagated_error * squash_derrivitive(weighted_sum);
    if ( verbose > 1 )
      cout << "\tnode " << node_number << " error = " << propagated_error << " * "
	   << squash_derrivitive(weighted_sum) << " = " << node_error << "\n";
  }

  return;

}

void node::adjust_weights( void ) {
  if ( verbose > 1 )
    cout << "\tAdjusting weights at node " << node_number << "\n";

  for ( int i=0; i<number_of_inputs; i++ ) {
    if ( verbose > 1 )
      cout << "\tweight[" << i << "] = " << weights[i] << " - "
	   << learning_rate << " * " << node_error << " * " << last_input[i] 
	   << " = ";

    weights[i] -= learning_rate * node_error * last_input[i];

    if ( verbose > 1 )
      cout << weights[i] << "\n";

  }
  return;
}

void node::dump_weights( void ) {
  if ( !p->getBool("BIAS") ) {
    cout << ", Node " << node_number << " (";
    for ( int i=0; i<number_of_inputs; i++ )
      cout << weights[i] << ", ";
    cout << "\b\b)";
  }
  else {
    cout << ", Node " << node_number << " (";
    for ( int i=0; i<number_of_inputs-1; i++ )
      cout << weights[i] << ", ";
    cout << "\b\b)";
    cout << ", bias = " << weights[number_of_inputs-1];
  }
  return;
}

int node::set_weights( int offset ) {
  if ( !weights )
    weights = new float [number_of_inputs];

  for ( int i=offset; i<offset+number_of_inputs; i++ )
    weights[i-offset] = p->getFloatArray("WEIGHTS")[i];

  return offset+number_of_inputs;
}
