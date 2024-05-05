enum TOPIC;
array[TOPIC] of int: hours;
int: m; % precedences
set of int: PREC = 1..m;
array[PREC] of TOPIC: before;
array[PREC] of TOPIC: after;
set of TOPIC: morning;
int: D; 
int: H;
set of int: DAY = 1..D;
set of DAY: HALF;

set of int: TIME = 0..D*H;
array[TOPIC] of var TIME: start;

/*** debug output ***
array[1..20] of string: key = ["*","+","$","%","@","#","&","=","-",":","a","b","c","d","e","f","g","h","i","j"];
output [ "\(TOPIC[i]) \"" ++ key[i] ++ "\", " | i in 1..n ] ++ ["HALF \".\"\n"];
output [ "| day " ++ show_int(6-H,d) | d in DAY ] ++ ["|\n"]; 
output [ let { int: j = arg_max([ fix(start[t]) <= i /\ fix(start[t]) + hours[t] > i | t in TOPIC ] ++ [ true]); } in
         if j <= card(TOPIC) then key[j] 
         elseif (i div H) + 1 in HALF /\ i mod H >= H div 2 then "."
         else " " endif
       | i in TIME ];  
*** end debug output ***/
include "disjunctive.mzn";

array[TOPIC] of var TIME: which_day;
constraint forall (t in TOPIC) (which_day[t] = ((start[t]) div 8) + 1);
constraint forall (p in PREC) (start[before[p]] + hours[before[p]] <= start[after[p]]);
constraint forall (t in TOPIC) (start[t] + hours[t] <= (D*H) /\ start[t] + hours[t] <= (which_day[t]*H));
constraint forall (morn in morning) (start[morn] mod H = 0);
constraint disjunctive (start ++ [(d-1)*H + H div 2 | d in HALF],hours ++ [H div 2 | d in HALF]);

% predicate culmulative ([        

solve maximize (min(start));
/*** grading output ***/       
output ["\nstart = \(start);\n, \(which_day)"];     