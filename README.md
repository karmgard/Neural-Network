# nn++

A reasonably complete feed-forward back-propagating neural network written 
from scratch in C++. This was mostly a learning exercise for working out the
math and the algorithm to build a network in software on-the-fly and save 
the structure. Using the network to categorize input vectors feeds the vectors 
into a threadpool for simultaneous calculations on multiple CPU/Hyper-threaded 
machines.

Depends: libthreadpool, libparameters, and libutilities
