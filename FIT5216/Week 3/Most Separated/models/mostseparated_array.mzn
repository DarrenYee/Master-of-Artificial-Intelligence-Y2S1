int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

set of int: P0 = 0..n;
array[1..k] of var P0: Sx;


var int: distance;
var int: actual_points = sum(i in 1..k)(Sx[i] != 0);

constraint forall(i in 1..k-1)(Sx[i] >= (Sx[i] != 0) + Sx[i+1]);

constraint actual_points <= k  /\ actual_points >= 1;

constraint distance = sum (i in 1..actual_points, j in 1..actual_points where i < j) (d[Sx[i],Sx[j]]);

var int: temp;
constraint temp = ((actual_points*(actual_points-1)) div 2);

constraint distance div temp >= l;

constraint value = sum(i in 1..actual_points)(v[Sx[i]]);

var int: value;
solve maximize value;
                                  
