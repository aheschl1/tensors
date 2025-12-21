# rtensors graphs

Graphs in rtensors are an IR between a models definition and execution. The goal is to add compilation with optimizations in the future.

## Nodes

A node is an atomic operation, such as addition, convolution, or activation functions. Nodes have a forward and backwards function, and can have weights. A node operates on a single backend at a time. 

## Graph

A graph is a collection of nodes, and it owns all tensors created and used as input for execution. A graph is executed on a single backend at a time.

## Group

A group is a graph and a node. A group holds a subgraph, and is the only exception to the rule that a graph operates on a single backend at a time. A group may switch the backend context for its subgraph. This enables efficient use of the remote backend, and swapping between CPU and GPU. This transitioning is to be transparent to the user.

## Graphs are lazy

You set an input to the graph; however, execution happens only when you call forward on the graph. This enables optimizations and fusing of operations in the future.
