int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

set of int: P0 = 0..n;
array[1..k] of var P0: Sx;

var int: distance;
var int: actual_points = sum (i in 1..k) (Sx[i] != 0);

constraint forall(i in 1..k -1) (Sx[i] >= Sx[i+1] + (Sx[i] != 0));

constraint distance = sum (i,j in 1..actual_points where i < j) (d[Sx[i],Sx[j]] >= l * ((actual_points *(actual_points -1)) div 2));

var int: value;
constraint value = sum (i in 1..actual_points) (v[Sx[i]]);
solve maximize value;
                                  
