array[int] of int: readings;
int: start;
int: n = length(readings);
set of int: HOUR = 1..n;
set of int: HOUR0 = 0..n;

enum CHOICE = {heat, strongly_heat, cool, strongly_cool, do_nothing};

array [CHOICE] of int : cost = [1,5,3,9,0];
array [CHOICE] of int : lower = [1,4,-2,-5,0];

array [HOUR] of var CHOICE: choice;

array [HOUR0] of var int : temp;

constraint temp[0] = start;

constraint forall (i in HOUR) (temp[i] = floor ((temp[i-1] + readings[i]) / 2) + lower[choice[i]]);

constraint forall (i in HOUR0) (temp[i] <= 30 /\ temp[i] >= 25);
 
var int : obj; 

constraint obj = sum (c in choice) (cost[c]);

solve minimize obj;

output ["temp = \(temp);\n"];

output ["choice = \(choice);\n"];








% constraint (j in 1..n, k in CHOICE) (choice[j] = adjustment[j]






