int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

                                  
var set of P: S;       % chosen set S
var int: value;        % objective value of chosen set

solve maximize value;    
         
