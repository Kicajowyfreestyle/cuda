# cuda

### Description
The repository above contains my first steps in CUDA.

### Already included
 * map
 * stencil
 * reduce

### How to use
Reduce
```bash
./reduce <arraySize> <operation type>
```
Where allowed operation types are: min, max, sum

### TODO
 * Map
   * pass arguments by command line
 * Stencil
   * pass arguments by command line
 * Reduce:
   * better division for reducing very big arrays (length>2^25)
 * General:
   * add some time measurement
   * enclose solution inside class
