int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

set of int: P0 = 0..n;
array[1..k] of var P0: Sx;

var int: distance;

constraint distance = sum (i in 1..k-1, j in i..k) (d[Sx[i],Sx[j]]);

var int: temp;
constraint temp = ((k*(k-1))/2);

constraint distance / ((k*(k-1))/2) >= l;

constraint forall (i in 1..k, j in 1..k where i!=j) (Sx[i] != Sx[j]);

var int: value;
solve maximize value;
                                  
