#include "global.h"
#include "layer.h"

layer::layer( int type, int inum, int nnum, int lnum ) {

  previous = next = NULL;
  layer_number     = lnum;
  number_of_inputs = inum;
  number_of_nodes  = nnum;
  layer_type       = type;

  // If we're going to use bias in the network, that's equivilent
  // to an extra input on each layer with a constant value of 1
  // and the weights determined by training become the bias values.
  bias = p->getBool("BIAS");

  if ( bias )
    number_of_inputs++;

  if ( verbose ) {
    if ( type == INPUT )
      cout << "Creating input layer with " << number_of_inputs << " inputs";
    else if ( type == HIDDEN )
      cout << "Creating layer #" << layer_number << " with " << number_of_inputs 
	   << " inputs to " << number_of_nodes << " nodes";
    else if ( type == OUTPUT )
      cout << "Creating output layer with " << number_of_inputs 
	   << " inputs to " << number_of_nodes << " nodes";
    cout << "\n";
  }

  if ( layer_type == INPUT ) {
    normalization = new float[number_of_inputs];
    input_offset  = new float[number_of_inputs];
    for ( int i=0; i<number_of_inputs; i++ ) {
      normalization[i] = 1.0f;
      input_offset[i]  = 0.0f;
    }
  }

  nodes       = new node* [number_of_nodes];
  output      = new float [number_of_nodes];
  for ( int i=0; i<number_of_nodes; i++ ) {
    output[i]      = 0.0f;
    nodes[i]       = new node( number_of_inputs, i, layer_type );
  }

 return;
}

layer::~layer(void) {

  if ( nodes ) {
    for ( int i=0; i<number_of_nodes; i++ )
 	delete nodes[i];
    delete [] nodes;
  }

  if ( output )
    delete [] output;

  if ( normalization )
    delete [] normalization;
  
  if ( input_offset )
    delete [] input_offset;

  if ( back_errors )
    delete [] back_errors;

  previous = next = NULL;

  return;
}

float * layer::input ( float * input_data ) {
  if ( verbose > 1 ) {
    cout << "Layer " << layer_number << " input (";
    for ( int i=0; i<number_of_inputs; i++ )
      cout << input_data[i] << ", ";
    cout << "\b\b)\n";
  }

  if ( bias )
    input_data[number_of_inputs-1] = 1.0f;

  get_node_outputs( input_data );

  if ( layer_type != OUTPUT )
    result = next->input(output);
  else
    result = output;

  return result;
}

void layer::get_node_outputs( float * input_data ) {

  if ( layer_type == INPUT ) {
    for ( int i=0; i<number_of_inputs; i++ )
      output[i] = input_data[i] * normalization[i] + input_offset[i];
  }
  else {
    for ( int i=0; i<number_of_nodes; i++ )
      output[i] = nodes[i]->input(input_data);
  }

  if ( layer_type != INPUT && verbose > 1 ) {
    cout << "Layer " << layer_number << " got (";

    for ( int i=0; i<number_of_nodes; i++ )
      cout << output[i] << ", ";
    cout << "\b\b) from the nodes\n";
  }
  return;
}

void layer::set_output_targets( float * targets ) {
  if ( layer_type != OUTPUT )
    return;

  for ( int i=0; i<number_of_nodes; i++ )
    nodes[i]->set_node_target(targets[i]);

  return;
}

void layer::adjust_weights( void ) {
  if ( verbose > 1 )
    cout << "Adjusting weights in layer " << layer_number << "\n";

  for ( int i=0; i<number_of_nodes; i++ )
    nodes[i]->adjust_weights();
  return;
}

float * layer::get_node_errors( void ) {

  // There's no way to know how big this array needs to be
  // in the instantiator, we've got to wait at least until
  // the linked list is being established. Better to wait 
  // until we need it then make it big enough
  if ( !back_errors ) {
    int n = previous->get_number_of_nodes();
    back_errors = new float [n];
    for ( int i=0; i<n; i++ )
      back_errors[i] = 0.0f;
  }

  // 1: Loop over the nodes in the previous (calling) layer
  for ( int i=0; i<previous->get_number_of_nodes(); i++ ) {

    // 2: For each node in this layer, get the node error
    //    and the weight that corresponds to the ith input
    //    (which should be the output from the ith node
    //    in the calling layer)
    
    if ( verbose > 1 )
      cout << "\tback_errors[" << i << "] = ";

    for ( int j=0; j<number_of_nodes; j++ ) {
      if ( verbose > 1 )
	cout <<  nodes[j]->get_weights()[i] << " * " << nodes[j]->get_node_error() << " + ";
      back_errors[i] += nodes[j]->get_weights()[i] * nodes[j]->get_node_error();
    }
    if ( verbose > 1 )
      cout << "\b\b = " << back_errors[i] << "\n";
  }

  // And send the resulting array back to the calling layer
  return back_errors;
}

void layer::propagate_errors( void ) {

  if ( layer_type == INPUT ) {
    next->adjust_weights();
    return;
  }

  if ( verbose > 1 ) {
    cout << "Propagating errors for layer " << layer_number;
    if ( layer_type == OUTPUT )
      cout << " (OUTPUT)\n";
    else if ( layer_type == HIDDEN )
      cout << " (HIDDEN)\n";
    else
      cout << " (UNKNOWN)\n";
  }

  float * propagated_errors;
  if ( next ) {
    propagated_errors = new float [next->get_number_of_nodes()];
    propagated_errors = next->get_node_errors();
  }
  else {
    propagated_errors = new float [number_of_nodes];
    for ( int i=0; i<number_of_nodes; i++ )
      propagated_errors[i] = 0.0f;
  }

  for ( int i=0; i<number_of_nodes; i++ )
    nodes[i]->calculate_error(propagated_errors[i]);

  if ( next )
    next->adjust_weights();

  return;
}


void layer::normalize_input( float * norms, float * offsets ) {

  if ( layer_type != INPUT )
    return;

  for ( int i=0; i<number_of_inputs-1; i++ ) {
    normalization[i] = norms[i];
    if ( offsets )
      input_offset[i] = offsets[i];
  }
  if ( bias ) {
    normalization[number_of_inputs-1] = 1.0f;
    input_offset[number_of_inputs-1]  = 0.0f;
  }
  else {
    normalization[number_of_inputs-1] = norms[number_of_inputs-1];
    input_offset[number_of_inputs-1]  = offsets[number_of_inputs-1];
  }

  return;
}

void layer::dump_weights(void) {

  cout << "Layer #" << layer_number;

  if ( layer_type == INPUT ) {

    if ( p->getBool("NORMALIZE") ) {
      int i = 0;
      for ( i=0; i<number_of_inputs-1; i++ )
	cout << ", normalization[" << i << "] = " << normalization[i] 
	     << ", input_offset[" << i << "] = " << input_offset[i];
      if ( !bias )
	cout << ", normalization[" << i << "] = " << normalization[i] 
	     << ", input_offset[" << i << "] = " << input_offset[i];
    }
    else {
      if ( bias )
	cout << ", " << number_of_inputs-1 << " input and 1 bias nodes";
      else
	cout << ", " << number_of_inputs << " input nodes";
    }
    cout << "\n";
    return;
  }
  
  for ( int i=0; i<number_of_nodes; i++ )
    nodes[i]->dump_weights();

  cout << "\n";

  return;
}

void layer::set_weights( int offset ) {
  if ( layer_type == INPUT && p->getBool("NORMALIZE") ) {
    for ( int i=0; i<number_of_inputs; i++ ) {
      if ( p->keyExists("NORMALIZATION") )
	normalization[i] = p->getFloatArray("NORMALIZATION")[i];
      if ( p->keyExists("OFFSET") )
	input_offset[i] = p->getFloatArray("OFFSET")[i];
    }
  }

  for ( int i=0; i<number_of_nodes; i++ )
    offset = nodes[i]->set_weights(offset);
  if ( this->next )
    next->set_weights(offset);

  return;
}

float * layer::get_weights(void) {
  if ( layer_type == INPUT )
    return NULL;

  if ( !weights )
    weights = new float [number_of_inputs * number_of_nodes];

  for ( int i=0; i<number_of_nodes; i++ ) {
    float *w = nodes[i]->get_weights();
    for ( int j=0; j<number_of_inputs; j++ )
      weights[i*number_of_inputs + j] = w[j];
  }
  return weights;
}
