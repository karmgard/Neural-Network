#include "global.h"
#include "network.h"
#include "squashlib.h"

/*************************************************************/
network *that;
void queue_data( void * data ) {
  
  float * in = (float *)data;
  float * result = that->input((float *)data);

  cout << "Running network: NN(";
  for ( int i=0; i<that->getNumberOfInputs(); i++ )
    cout << in[i] << ", ";
  cout << "\b\b) = ";

  for ( int i=0; i<that->getNumberOfOutputs(); i++ )
    cout << "(" << result[i] << ", ";
  cout << "\b\b)\n";

  fflush(0);
  return;
}
/*************************************************************/

network::network(void) {

  if ( !p->getBool("TRAIN") && p->keyExists("NETWORK_FILE") ) {
    if ( verbose )
      cout << "Reading network file " << p->getString("NETWORK_FILE") << "\n";
    p->readFile(p->getString("NETWORK_FILE"));
  }

  parse_topology();

  first = new layer(INPUT, number_of_inputs);
  layer *temp = first;
  for ( int i=1; i<number_of_layers-1; i++ ) {
    temp->set_next( new layer(HIDDEN         /* layer type */, 
			      topology[i-1], /* number of inputs to this layer */
			      topology[i],   /* number of nodes in the layer */
			      i              /* Layer ID number */
			      )
		    );
    temp->get_next()->set_previous(temp);
    temp = temp->get_next();
  }
  last = new layer(OUTPUT, 
		   topology[number_of_layers-2], 
		   topology[number_of_layers-1], 
		   number_of_layers-1
		   );
  last->set_previous(temp);
  last->get_previous()->set_next(last);

  initialize_squashlib( p->getString("SQUASH_FUNCTION"), 
			p->getString("ERROR_FUNCTION"), 
			p->getFloat("TEMPERATURE") );

  if ( verbose ) 
    cout << "Network initialized\n";

  that = this;

  return;
}

network::~network(void) {

  if ( verbose )
    cout << "Cleaning up\n";

  layer * temp = last;
  while ( temp ) {
    layer * temp2 = temp;
    temp = temp->get_previous();
    delete temp2;
  }

  if ( topology )
    delete [] topology;
  return;
}

void network::parse_topology(void) {
  number_of_layers  = p->getInt("TOPOLOGY_SIZE");
  number_of_inputs  = p->getUIntArray("TOPOLOGY")[0];
  number_of_outputs = p->getUIntArray("TOPOLOGY")[number_of_layers-1];

  topology = new int [number_of_layers];
  for ( int i=0; i<number_of_layers; i++ )
    topology[i] = p->getUIntArray("TOPOLOGY")[i];

  if ( verbose ) {
    cout << "Creating a network with " << number_of_layers << " layers: "
	 << number_of_inputs << " inputs, " << number_of_outputs
	 << " outputs, and " << number_of_layers-2 << " hidden layers\n";
  }
  return;
}

int network::normalize ( string input_file ) {

  float * scale = new float [number_of_inputs];
  float * max   = new float [number_of_inputs];
  float * min   = new float [number_of_inputs];

  for ( int i=0; i<number_of_inputs; i++ ) {
    max[i] = scale[i] = 0.0f;
    min[i] = 999999.0f;
  }

  ifstream infile;
  infile.open(input_file);
  if ( !infile.is_open() ) {
    cerr << "Unable to open input file " << input_file << "\n";
    return 2;
  }

  if ( verbose )
    cout << "Normalizing inputs from " << input_file << "\n";

  string line;
  string delimiter(" ");

  while ( getline( infile, line ) ) {
    size_t pos = 0;
    string token;
    int counter = 0;
 
    while ((pos = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, pos);

      if ( counter < number_of_inputs ) {
	float value = fabs(atof(token.c_str()));
	if ( value > max[counter] )
	  max[counter] = value;
	else if ( value < min[counter] )
	  min[counter] = value;
      }
      counter++;
      line.erase(0, pos + delimiter.length());
    }
    if ( counter < number_of_inputs ) {
      float value = fabs(atof(line.c_str()));
	if ( value > max[counter] )
	  max[counter] = value;
	else if ( value < min[counter] )
	  min[counter] = value;
    }
  }

  for ( int i=0; i<number_of_inputs; i++ ) {
    if ( max[i] - min[i] != 0.0f )
      scale[i] = 1.0f/(max[i] - min[i]);
    if ( verbose )
      cout << "Input " << i << ": max = " << max[i] << ", min = " << min[i] 
	   << ", scale = " << scale[i] << "\n";
  }

  first->normalize_input(scale, min);

  delete [] min;
  delete [] max;
  delete [] scale;

  return 0;
}

int network::train_network( void ) {

  float input[number_of_inputs];
  float output[number_of_outputs];

  if ( p->getBool("NORMALIZE") )
    normalize(p->getString("TRAINING_FILE"));

  ifstream training;
  training.open(p->getString("TRAINING_FILE"));
  if ( !training.is_open() ) {
    cerr << "Unable to open training file " << p->getString("TRAINING_FILE") << "\n";
    return 2;
  }

  string line;
  string delimiter(" ");
 
  uint number_of_cycles = p->getUInt("TRAINING_CYCLES");
  uint cycle = 0;
  float training_limit = p->getFloat("TRAINING_LIMIT");
  int number_of_vectors = 0;
  float error = 0.0f;

  for ( uint i=0; i<number_of_cycles; i++ ) {
    while ( getline( training, line ) ) {
      size_t pos = 0;
      string token;
      int counter = 0;
 
      while ((pos = line.find(delimiter)) != std::string::npos) {
	token = line.substr(0, pos);

	if ( counter < number_of_inputs )
	  input[counter] = atof(token.c_str());
	else
	  output[counter-number_of_inputs] = atof(token.c_str());
	counter++;

	line.erase(0, pos + delimiter.length());
      }
      output[counter-number_of_inputs] = atof(line.c_str());
      error += train(input, output);
      number_of_vectors++;
    }

    cout << "Convergence check: cycle " << ++cycle << " error = " << error << "\n\n";
    if ( error < training_limit * number_of_vectors )
      break;

    error = 0.0f;
    number_of_vectors = 0;

    // Rewind the training file
    training.clear();
    training.seekg(0);
  }

  training.close();

  dump_weights();

  // Save the network structure if there's a filename given
  if ( p->keyExists("SAVE_TRAINING_FILE") )
    save_network(p->getString("SAVE_TRAINING_FILE"));


  return 0;
}

float network::train( float * input_data, float * target ) {

  float * result = input( input_data );

  cout << "NN(";
  for ( int i=0; i<number_of_inputs; i++ )
    cout << input_data[i] << ", ";
  cout << "\b\b) = (";
  for ( int i=0; i<number_of_outputs; i++ )
    cout << result[i] << ", ";
  cout << "\b\b) vs. (";
  for ( int i=0; i<number_of_outputs; i++ )
    cout << target[i] << ", ";
  cout << "\b\b)\n";

  layer * l = last;
  l->set_output_targets( target );

  while ( l ) {
    l->propagate_errors();
    l = l->get_previous();
  }

  // Calculate a rough total error for this vector
  float error = 0.0f;
  for ( int i=0; i<number_of_outputs; i++ )
    error += squash_error( result[i], target[i] );

  return error;

}
/********
void network::input( void * inPtr ) {

  float * input_data = (float *)inPtr;
  float * result = input(input_data);

  for ( int i=0; i<number_of_outputs; i++ )
    cout << "(" << result << ", ";
  cout << "\b\b)\n";

  return;
}
*********/

float * network::input( float * input_data ) {
  if ( verbose > 1 ) {
    cout << "Sending (";
    for (int i=0; i<number_of_inputs; i++)
      cout << input_data[i] << ", ";
    cout << "\b\b) into the network\n";
  }

  if ( first )
    result = first->input( input_data );
  else
    cerr << "First layer not defined!\n";

  return result;
}

void network::dump_weights(void) {
  layer * l = first;
  while ( l ) {
    l->dump_weights();
    l = l->get_next();
  }
  cout << "\b \n";
  return;
}

void network::set_weights(void) {
  if ( !p->keyExists("WEIGHTS") ) {
    cerr << "Weights array does not exist\n";
    return;
  }

  if ( first ) 
    first->set_weights(0);
  else {
    cerr << "Cannot find head of the linked list!\n";
    exit(1);
  }
  return;
}

void network::save_network( string fileName ) {

  parameters *n = new parameters();
  n->setUIntArray ("TOPOLOGY",        (uint *)topology, (uint)number_of_layers );
  n->setString    ("SQUASH_FUNCTION", p->getString("SQUASH_FUNCTION")          );
  n->setString    ("ERROR_FUNCTION",  p->getString("ERROR_FUNCTION")           );
  n->setFloat     ("LEARNING_RATE",   p->getFloat("LEARNING_RATE")             );
  n->setFloat     ("TEMPERATURE",     p->getFloat("TEMPERATURE")               );
  n->setBool      ("BIAS",            p->getBool("BIAS")                       );
  n->setBool      ("NORMALIZE",       p->getBool("NORMALIZE")                  );
  uint size = 0;

  if ( p->keyExists("WEIGHTS_SIZE") )
    size = p->getUInt("WEIGHTS_SIZE");
  else {
    for ( int i=1; i<number_of_layers; i++ )
      size += (topology[i-1]+(uint)p->getBool("BIAS")) * topology[i];
  }
  float weights[size];
  
  float *w;
  int offset = 0;

  layer *l = first;
  while ( l ) {
    w = l->get_weights();
    for ( int i=0; i<l->get_number_of_weights(); i++ )
      weights[offset+i] = w[i];
    offset += l->get_number_of_weights();
    l = l->get_next();
  }

  n->setFloatArray("WEIGHTS", weights, size);
  n->setFloatArray("NORMALIZATION", first->get_norms(), number_of_inputs);
  n->setFloatArray("OFFSET", first->get_offsets(), number_of_inputs);
  
  if ( verbose )
    cout << "Saving to " << fileName << "\n";
  n->save(fileName);
  delete n;

  if ( verbose )
    cout << "\n";

  return;
}
