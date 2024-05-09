enum TOPIC;
int: n = card(TOPIC);
array[TOPIC] of int: hours;
int: m; % precedences
set of int: PREC = 1..m;
array[PREC] of TOPIC: before;
array[PREC] of TOPIC: after;
set of TOPIC: morning;
int: D; 
int: H;
constraint assert(H mod 2 = 0,"Hours per day H must be even\n");
set of int: DAY = 1..D;
set of DAY: HALF;

set of int: HOUR = 0..H-1;
array[TOPIC] of var DAY: day;
array[TOPIC] of var HOUR: hour;

% constraint forall (p in PREC) (start[before[p]] + hours[before[p]] <= start[after[p]]);
% constraint forall (t in TOPIC) (start[t] + hours[t] <= (D*H));
% constraint forall (morn in morning) (start[morn] mod H = 0);
% constraint disjunctive (start ++ [(d-1)*H + H div 2 | d in HALF],hours ++ [H div 2 | d in HALF]);

include "diffn.mzn";
include "cumulative.mzn";
constraint diffn (hour ++[ H div 2 | d in HALF],day ++[d | d in HALF],hours ++ [H div 2 | d in HALF],[1|t in TOPIC]++[1 | d in HALF]);
constraint forall (i in TOPIC) (hour[i] + hours[i] <= H);
constraint forall (p in PREC) (day[before[p]] < day[after[p]] \/(day[before[p]] < day[after[p]] /\ hour[before[p]] + hours[before[p]] <= hour[after[p]]));
constraint forall (morn in morning) (hour[morn] = 0);

solve maximize (min(day));
%% output for grading       
output ["day = \(day);\nhour = \(hour);\n"];       
       
       