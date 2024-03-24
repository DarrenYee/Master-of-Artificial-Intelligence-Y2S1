set of int: POS = 1..4;
enum REINDEER = { Lancer, Quentin, Ezekiel, Rudy };
array[POS] of var REINDEER: y;

include "alldifferent.mzn";

constraint (POS[y[Lancer]] != POS[y[Ezekiel]]- 1) /\ (POS[y[Lancer]] != POS[y[Ezekiel]] + 1); 
constraint (POS[y[Rudy]] < POS[y[Lancer]]) \/ (POS[y[Rudy]] < POS[y[Quentin]]); 

constraint alldifferent (y);

output ["y = \(y);\n"];


  