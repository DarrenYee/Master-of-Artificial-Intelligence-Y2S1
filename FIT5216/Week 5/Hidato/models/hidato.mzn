% hidato puzzle

int: n; % no of ROWS
set of int: ROW = 1..n;
int: m; % no of COLS
set of int: COL = 1..m;

set of int: CLUE = 0..m*n;
array[ROW,COL] of CLUE: clue;

set of int: NUM = 1..m*n;
array[ROW,COL] of var NUM: x;

set of int: ROWCOL = 1..m*n;
array [NUM] of var ROWCOL: y;

constraint forall(r in ROW, c in COL where clue[r,c] != 0) (x[r,c] = clue[r,c]);

% constraint forall (r in ROW, c in COL where x[r,c] != m*n) ((x[r,c+1] = x[r,c] + 1) \/ (x[r-1,c+1] = x[r,c] + 1) \/ (x[r-1,c] = x[r,c] + 1) \/ (x[r-1,c-1] = x[r,c] + 1) \/ (x[r,c-1] = x[r,c] + 1) \/ (x[r+1,c-1] = x[r,c] + 1) \/ (x[r+1,c] = x[r,c] + 1) \/ (x[r+1,c+1] = x[r,c] + 1));

% constraint forall (num in 1..((m*n) -1)) ((y[n+1] = y[n] + 1) \/ (y[n+1] = y[n] - 1) \/ (y[n+1] = (((y[n] div 10) mod 10 + 1) mod 10) *10 + y[n] mod 10) \/ (y[n+1] = (((y[n] div 10) mod 10 + 1) mod 10) *10 + y[n] mod 10 + 1) \/ (y[n+1] = (((y[n] div 10) mod 10 + 1) mod 10) *10 + y[n] mod 10 + 2) \/ (y[n+1] = (((y[n] div 10) mod 10 - 1) mod 10) *10 + y[n] mod 10) \/ (y[n+1] = (((y[n] div 10) mod 10 - 1) mod 10) *10 + y[n] mod 10 + 1) \/ (y[n+1] = (((y[n] div 10) mod 10 - 1) mod 10) *10 + y[n] mod 10 + 2));

include "alldifferent.mzn";
include "globals.mzn";

constraint inverse(y,
[x[r,c]|r in ROW, c in COL]);


%% output for debugging                
% int: d = ceil(log10(m*n+1));
% output [ show_int(d,x[r,c]) ++ " " ++ if c = m then "\n" else "" endif
%        | r in ROW, c in COL ];                 

%% output for grading
% output ["x = array2d(ROW,COL,\(x));\n"];