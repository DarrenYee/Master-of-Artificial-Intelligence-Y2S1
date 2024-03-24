enum DIRN = {n,e,s,w};

var DIRN: A;
var DIRN: B;
var DIRN: C;
var DIRN: D;
var DIRN: E;
var DIRN: F;


array [1..6] of var DIRN: searchers = [A,B,C,D,E,F];
% Domains = set of all possible inputs, Co-domains = set of all possible combinations of outcomes

constraint forall (j in DIRN) (exists (i in searchers) (searchers[i] = DIRN[j]));

constraint (A = s \/ A = n) -> (B != s \/ B != n); 

constraint ((C = s) -> (D = n)) /\ ((C = n) -> (D = s)) /\ ((C = w) -> (D = e)) /\ ((C = e) -> (D = w));

constraint A != E /\ B!= F /\ C!= E;
 
solve satisfy;