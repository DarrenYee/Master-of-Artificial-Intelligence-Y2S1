int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

set of int: P0 = 0..n;
array[1..k] of var P0: Sx;


var int: distance;
var int: actual_points;


constraint actual_points <= k  /\ actual_points >= 1;

constraint distance = sum (i in 1..actual_points, j in 1..actual_points where i < j) (d[Sx[i],Sx[j]]);


var int: temp;
constraint temp = ((k*(k-1))/2);

% constraint distance / ((k*(k-1))/2) >= l;

constraint forall (i in 1..k-1) (Sx[i] >= (Sx[i]!=0)+Sx[i+1]);

var int: value;
solve maximize value;
                                  
