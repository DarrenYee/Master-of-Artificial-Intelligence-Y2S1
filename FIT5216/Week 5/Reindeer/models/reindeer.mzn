set of int: POS = 1..4;
enum REINDEER = { Lancer, Quentin, Ezekiel, Rudy };
array[REINDEER] of var POS: x;
include "alldifferent.mzn";

constraint (x[Lancer] != x[Ezekiel] - 1) /\ (x[Lancer] != x[Ezekiel] + 1); 
constraint (x[Rudy] < x[Lancer]) \/ (x[Rudy] < x[Quentin]); 
constraint alldifferent (x);

output ["x = array1d(REINDEER,\(x));\n"];
  