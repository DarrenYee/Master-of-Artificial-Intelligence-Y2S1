array[int] of int: readings;
int: start;
int: n = length(readings);
set of int: HOUR = 1..n;
set of int: HOUR0 = 0..n;
array [1..n+1] of var 0..infinity : temp;

constraint temp[1] = start;
constraint forall (i in 2..n+1) (temp[i] = floor ((temp[i-1] + readings[i-1]) / 2));


output ["temp = \(temp);\n"];
