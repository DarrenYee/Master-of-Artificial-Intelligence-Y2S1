% nurse rostering project
% STAGE A data
enum NURSE; % set of nurses to roster
enum SHIFT = { MORN, DAY, EVEN, NIGH, OFF };
int: nday;     % number of days for roster
set of int: DAYS = 1..nday; 
array[NURSE,DAYS] of bool: rostered_off; % nurse is guaranteed an off shift
% STAGE B data
int: maxweek; % maximum working shifts in any 7 day period
int: maxnightfort; % maximum night shifts in any 14 day period
int: minfort; % minimum number of working shifts in any 14 day period
array[SHIFT,DAYS] of int: minshift; % minimum number of nurses rostered on each shift
set of int: DAYSPERWEEK = 0..6;
set of int: DAYSPERFORT = 0..13;
% STAGE C data
array[NURSE] of int: shift_cost;  
% STAGE D data
enum WARD;  % set of wards to roster
WARD: dummy; % dummy ward
array[WARD,DAYS] of int: minward; % minimum number of nurses rostered to each ward
constraint forall(d in DAYS)(assert(minward[dummy,d] = 0, "Dummy ward has non zero minimum on day \(d)\n"));
int: maxward; % maximum number of wards any nurse can work in the roster period
% STAGE E data
enum SKILL;
SKILL: senior;                         % which skill is senior
array[NURSE] of set of SKILL: skill;   % skills that each nurse has
array[WARD] of set of SKILL: required; % particular skills for each ward
opt WARD: emergency; % emergency ward if its being rostered
enum SLICE = { PREDAWN, AM, PM, LATE };
% coverage MORNING = {PREDAWN, AM}, 
%          DAY = {AM,PM} 
%          EVENING = {PM, LATE}, 
%          NIGHT {LATE, PREDAWN of the next day} 

%%%% DECISIONS
array[NURSE,DAYS] of var SHIFT: sh; % shift for each nurse on each day
var 0..nday*sum(shift_cost): total_cost;
array[NURSE,DAYS] of var WARD: wd; % ward rostered for each nurse
array[WARD,DAYS] of var int: count_wd; % ward rostered for each nurse



% MORN = no rules
% DAY = DAY -> DAY then no rules
% EVEN = EVEN -> all except for MORN
% NIGH = No NIGH -> NIGH -> NIGH, NO NIGH -> DAY or MORN
% OFF = no OFF -> OFF

include "globals.mzn";
include "regular.mzn";  

      
array[1..10,SHIFT] of 0..10: trans =
      [|1,2,3,4,5  %MORN(1)
      | 0,8,0,0,0  %DAY(2)
      | 0,2,3,4,5  %EVEN(3)
      | 0,0,3,6,5  %NIGH1(4)
      | 1,2,3,4,9  %OFF1(5)
      | 0,0,3,7,5  %NIGH2(6)
      | 0,0,3,0,5  %NIGH3(7)
      | 1,8,3,4,5  %DAY2(8)
      | 1,2,3,4,0  %OFF2(9)
      | 1,2,3,4,5|]; %DUMMY(10)     

constraint forall (n in NURSE, day in DAYS) (sh[n,day] = OFF <-> wd[n,day] = dummy);

constraint forall (n in NURSE, day in DAYS) (rostered_off[n,day] -> (sh[n,day] = OFF));

constraint forall(n in NURSE) (regular ([sh[n,day] |day in DAYS],10,5,trans,10,{1,2,3,4,5,6,7,8,9,10}));


% Each Shift s for each d has at least minshift [s,d]
% No more than maxweek shifts (excluding off) in 7 day period
% No less than minfort in 14 day period
% No more than maxnightfort NIGH in any 14 day perdod


constraint forall (day in DAYS) 
  (global_cardinality (
    [sh[n,day] | n in NURSE],
    [MORN, DAY, EVEN, NIGH, OFF],
    [minshift[MORN,day],minshift[DAY,day],minshift[EVEN,day],minshift[NIGH,day],minshift[OFF,day]],
    [card(NURSE),card(NURSE),card(NURSE),card(NURSE),card(NURSE)]
));

array [NURSE,DAYS] of var 0..1: sh_off;
array [NURSE,DAYS] of var 0..1: sh_nigh;

constraint forall (n in NURSE, day in DAYS) (sh_off[n,day] = bool2int(sh[n,day] != OFF) /\ sh_nigh[n,day] = bool2int(sh[n,day] = NIGH));

constraint forall (n in NURSE) (sliding_sum(0, maxweek, 7,[sh_off[n, day] |day in DAYS]) /\ sliding_sum(minfort, nday, 14,[sh_off[n, day] | day in DAYS]) /\ sliding_sum(0, maxnightfort, 14,[sh_nigh[n, day] | day in DAYS]));

array[NURSE] of var int: total_shifts;

constraint total_cost = (sum(n in NURSE,d in DAYS) (sh_off[n,d]* shift_cost[n]));

constraint forall (w in WARD, day in DAYS) (count_wd[w,day] = (sum(n in NURSE) (wd[n,day] = w)) /\ count_wd[w,day] >= minward[w,day]);

array [NURSE] of var 1..maxward : distinct_wards;

constraint forall (n in NURSE) (distinct_wards[n] = nvalue([wd[n,day]|day in DAYS]));

% If wd = ward then ward's speciality has to be on duty
set of int: skill_max = 0..card(SKILL) +1;
array [NURSE, DAYS] of var skill_max: testing;
constraint forall(n in NURSE,day in DAYS)(testing[n,day] =  card(skill[n] intersect required[wd[n,day]]) + (if (required[wd[n,day]] = {}) then 1 else 0 endif));
array [WARD,DAYS] of var 0..card(NURSE): testing1; 

% constraint forall (n in NURSE,day in DAYS) (testing[n,day] > 0 );

constraint forall (w in WARD, day in DAYS) (if (minward[w,day] > 0) then ((testing1[w,day] = (sum (n in NURSE) (bool2int(wd[n,day] = w)*testing[n,day]))) > 0) endif);

% % FOr each slice, there must be a senior

% If emergency = True, 




solve minimize total_cost;