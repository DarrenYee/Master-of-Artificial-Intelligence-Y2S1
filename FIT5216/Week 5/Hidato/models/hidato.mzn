% hidato puzzle

int: n; % no of ROWS
set of int: ROW = 1..n;
int: m; % no of COLS
set of int: COL = 1..m;

set of int: CLUE = 0..m*n;
array[ROW,COL] of CLUE: clue;

set of int: NUM = 1..m*n;
array[ROW,COL] of var NUM: x;

set of int: POS = 1..n*m;
array [NUM] of var POS: y;

predicate adjacent (var POS: a, var POS: b) = 
          let { var ROW: ra = (a-1) div m + 1;
                var COL: ca = (a-1) mod m + 1;
                var ROW: rb = (b-1) div m + 1;
                var COL: cb = (b-1) mod m + 1;
          } in
          abs (ra-rb) <= 1 /\ abs (ca-cb) <= 1;

constraint forall(r in ROW, c in COL) (if clue[r,c] > 0 then x[r,c] = clue[r,c] endif);

include "inverse.mzn";
constraint inverse(array1d(x),y);
constraint forall (i in 1..m*n-1) (adjacent(y[i],y[i+1]));





%% output for debugging                
% int: d = ceil(log10(m*n+1));
% output [ show_int(d,x[r,c]) ++ " " ++ if c = m then "\n" else "" endif
%        | r in ROW, c in COL ];                 

%% output for grading
% output ["x = array2d(ROW,COL,\(x));\n"];